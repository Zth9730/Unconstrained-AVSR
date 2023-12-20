from typing import Optional, Tuple, Union
from dataclasses import dataclass
import os
import logging
import warnings
import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder as HFWhisperEncoder
from transformers.models.whisper.modeling_whisper import WhisperModel as HFWhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer as HFWhisperEncoderLayer
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration as HFWhisperForConditionalGeneration
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput
)
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperFlashAttention2,
    WHISPER_ATTENTION_CLASSES,
    _CONFIG_FOR_DOC,
    WHISPER_INPUTS_DOCSTRING,
    shift_tokens_right
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,   
)

logger = logging.getLogger(__name__)

class WhisperEncoderLayer(HFWhisperEncoderLayer):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.embed_dim = config.d_model
        self.if_cross_attn = config.if_cross_attn
        if self.if_cross_attn:
            self.cross_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
            )
            self.corss_attn_layer_norm = nn.LayerNorm(self.embed_dim)
            
    def forward(
        self,
        image_features: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        image_features = image_features.unsqueeze(1)
        if not self.if_cross_attn:
            hidden_states = torch.cat([image_features, hidden_states], dim=1)

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.if_cross_attn:
            residual = hidden_states
            hidden_states = self.corss_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, _ = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=image_features,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class WhisperEncoder(HFWhisperEncoder):
    """
    overwrite forward to support attention_mask
    overwrite from_pretrained to support split encoder parameters from pretrained WhisperModel
    """
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        
    def forward(
        self,
        image_features,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        image_features,
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class WhisperModel(HFWhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        image_features: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)
            encoder_outputs = self.encoder(
                image_features,
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class WhisperForConditionalGeneration(HFWhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel(config)
        
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        image_features: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            image_features,
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    # def generate(
    #     self,
    #     input_features: Optional[torch.Tensor] = None,
    #     generation_config=None,
    #     logits_processor=None,
    #     stopping_criteria=None,
    #     prefix_allowed_tokens_fn=None,
    #     synced_gpus=False,
    #     return_timestamps=None,
    #     task=None,
    #     language=None,
    #     is_multilingual=None,
    #     prompt_ids: Optional[torch.Tensor] = None,
    #     num_segment_frames: Optional[int] = None,
    #     return_token_timestamps: Optional[bool] = None,
    #     return_segments: bool = False,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     time_precision: int = 0.02,
    #     return_dict_in_generate: Optional[bool] = None,
    #     **kwargs,
    # ):
    #     """
    #     Transcribes or translates passed mel input features to a sequence of token ids.

    #     <Tip warning={true}>

    #     Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    #     model's default generation configuration. You can override any `generation_config` by passing the corresponding
    #     parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    #     For an overview of generation strategies and code examples, check out the [following
    #     guide](./generation_strategies).

    #     </Tip>

    #     Parameters:
    #         inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
    #             The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
    #             method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
    #             should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
    #             `input_ids`, `input_values`, `input_features`, or `pixel_values`.
    #         generation_config (`~generation.GenerationConfig`, *optional*):
    #             The generation configuration to be used as base parametrization for the generation call. `**kwargs`
    #             passed to generate matching the attributes of `generation_config` will override them. If
    #             `generation_config` is not provided, the default will be used, which had the following loading
    #             priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
    #             configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
    #             default values, whose documentation should be checked to parameterize generation.
    #         logits_processor (`LogitsProcessorList`, *optional*):
    #             Custom logits processors that complement the default logits processors built from arguments and
    #             generation config. If a logit processor is passed that is already created with the arguments or a
    #             generation config an error is thrown. This feature is intended for advanced users.
    #         stopping_criteria (`StoppingCriteriaList`, *optional*):
    #             Custom stopping criteria that complement the default stopping criteria built from arguments and a
    #             generation config. If a stopping criteria is passed that is already created with the arguments or a
    #             generation config an error is thrown. This feature is intended for advanced users.
    #         prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
    #             If provided, this function constraints the beam search to allowed tokens only at each step. If not
    #             provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
    #             `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
    #             on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
    #             for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
    #             Retrieval](https://arxiv.org/abs/2010.00904).
    #         synced_gpus (`bool`, *optional*, defaults to `False`):
    #             Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
    #         return_timestamps (`bool`, *optional*):
    #             Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
    #         task (`str`, *optional*):
    #             Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
    #             will be updated accordingly.
    #         language (`str`, *optional*):
    #             Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
    #             find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
    #         is_multilingual (`bool`, *optional*):
    #             Whether or not the model is multilingual.
    #         prompt_ids (`torch.Tensor`, *optional*):
    #             Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
    #             provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
    #             transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
    #             correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
    #         return_token_timestamps (`bool`, *optional*):
    #             Whether to return token-level timestamps with the text. This can be used with or without the
    #             `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
    #             words.
    #         return_segments (`bool`, *optional*, defaults to `False`):
    #             Whether to additionally return a list of all segments. Note that this option can only be enabled
    #             when doing long-form transcription.
    #         attention_mask (`torch.Tensor`, *optional*):
    #             `attention_mask` needs to be passed when doing long-form transcription using a batch size > 1.
    #         time_precision (`int`, *optional*, defaults to 0.02):
    #             The duration of output token in seconds. *E.g.* 0.02 means that a generated token on average accounts
    #             for 20 ms.
    #         return_dict_in_generate (`bool`, *optional*, defaults to `False`):
    #             Whether or not to return a [`~utils.ModelOutput`] instead of just returning the generated tokens.
    #             Note that when doing long-form transcription, `return_dict_in_generate` can only be enabled when
    #             `return_segments` is set True. In this case the generation outputs of each segment is added to each
    #             segment.
    #         kwargs (`Dict[str, Any]`, *optional*):
    #             Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
    #             forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
    #             specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    #     Return:
    #         [`~utils.ModelOutput`] or `torch.LongTensor` or `Dict[str, Any]`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
    #         or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor` or a dict of segments when `return_segments=True`.

    #             If the passed input is > 30 seconds / > 3000 mel input features and `return_segments=True` then a dictionary of generated sequence ids, called `sequences` and a list of each generated segment is returned.

    #             else if the passed input is <= 30 seconds / >= 3000 mel input features, the possible [`~utils.ModelOutput`] types are:

    #                 - [`~generation.GreedySearchEncoderDecoderOutput`],
    #                 - [`~generation.SampleEncoderDecoderOutput`],
    #                 - [`~generation.BeamSearchEncoderDecoderOutput`],
    #                 - [`~generation.BeamSampleEncoderDecoderOutput`]

    #             else only the generated output sequence ids are returned.

    #     Example:

    #     - *Longform transcription*: To transcribe or translate audios longer than 30 seconds, process the audio files without truncation and pass all mel features at once to generate.

    #     ```python
    #     >>> import torch
    #     >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    #     >>> from datasets import load_dataset, Audio

    #     >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    #     >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    #     >>> model.cuda()

    #     >>> # load audios > 30 seconds
    #     >>> ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
    #     >>> # resample to 16kHz
    #     >>> ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    #     >>> # take first 8 audios and retrieve array
    #     >>> audio = ds[:8]["audio"]
    #     >>> audio = [x["array"] for x in audio]

    #     >>> # make sure to NOT truncate the input audio, to return the `attention_mask` and to pad to the longest audio
    #     >>> inputs = processor(audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
    #     >>> inputs = inputs.to("cuda", torch.float32)

    #     >>> # transcribe audio to ids
    #     >>> generated_ids = model.generate(**inputs)

    #     >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    #     >>> transcription[0]
    #     ' Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile!'
    #     ```

    #     - *Shortform transcription*: If passed mel input features are < 30 seconds, the whole audio will be transcribed with a single call to generate.

    #     ```python
    #     >>> import torch
    #     >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    #     >>> from datasets import load_dataset

    #     >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    #     >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    #     >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    #     >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    #     >>> input_features = inputs.input_features

    #     >>> generated_ids = model.generate(inputs=input_features)

    #     >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #     >>> transcription
    #     ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
    #     ```

    #     """

    #     if "inputs" in kwargs:
    #         input_features = kwargs.pop("inputs")
    #         warnings.warn(
    #             "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
    #             FutureWarning,
    #         )

    #     return_dict_in_generate = (
    #         return_dict_in_generate
    #         if return_dict_in_generate is not None
    #         else self.generation_config.return_dict_in_generate
    #     )

    #     if generation_config is None:
    #         generation_config = copy.deepcopy(self.generation_config)

    #     input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
    #     if num_segment_frames is None:
    #         num_segment_frames = input_stride * self.config.max_source_positions

    #     # 1. Check whether we're in shortform or longform mode
    #     if input_features is not None:
    #         total_input_frames = input_features.shape[-1]
    #     elif "encoder_outputs" in kwargs:
    #         encoder_outputs_shape = (
    #             kwargs["encoder_outputs"][0].shape
    #             if isinstance(kwargs["encoder_outputs"], BaseModelOutput)
    #             else kwargs["encoder_outputs"].shape
    #         )
    #         total_input_frames = encoder_outputs_shape[1] * input_stride
    #     else:
    #         raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `generate`.")

    #     is_shortform = total_input_frames <= num_segment_frames

    #     # 2. Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
    #     if return_timestamps is True:
    #         if not hasattr(generation_config, "no_timestamps_token_id"):
    #             raise ValueError(
    #                 "You are trying to return timestamps, but the generation config is not properly set. "
    #                 "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
    #                 "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
    #             )
    #         generation_config.return_timestamps = return_timestamps
    #     elif not is_shortform:
    #         if return_timestamps is False:
    #             raise ValueError(
    #                 "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
    #                 "requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features."
    #             )

    #         if not hasattr(generation_config, "no_timestamps_token_id"):
    #             raise ValueError(
    #                 "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
    #                 "requires the generation config to have `no_timestamps_token_id` correctly. "
    #                 "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
    #                 "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
    #                 "or make sure to pass no more than 3000 mel input features."
    #             )

    #         logger.info("Setting `return_timestamps=True` for long-form generation.")
    #         generation_config.return_timestamps = True
    #     else:
    #         generation_config.return_timestamps = False

    #     # 3. Make sure to correctly set language-related parameters
    #     if is_multilingual is not None:
    #         if not hasattr(generation_config, "is_multilingual"):
    #             raise ValueError(
    #                 "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
    #                 "to `generate`. Please update the generation config as per the instructions "
    #                 "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
    #             )
    #         generation_config.is_multilingual = is_multilingual

    #     if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
    #         if task is not None or language is not None:
    #             raise ValueError(
    #                 "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
    #                 "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
    #             )

    #     if language is not None:
    #         if not hasattr(generation_config, "lang_to_id"):
    #             raise ValueError(
    #                 "The generation config is outdated and is thus not compatible with the `language` argument "
    #                 "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
    #                 "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
    #             )
    #         language = language.lower()
    #         generation_config.language = language
    #     if task is not None:
    #         if not hasattr(generation_config, "task_to_id"):
    #             raise ValueError(
    #                 "The generation config is outdated and is thus not compatible with the `task` argument "
    #                 "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
    #                 "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
    #             )
    #         generation_config.task = task

    #     # 4. Add forced decoder ids depending on passed `language`, `task`,`prompt_ids`, `return_token_timestamps` and `return_timestamps`
    #     forced_decoder_ids = None
    #     # Legacy code for backward compatibility
    #     if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
    #         forced_decoder_ids = self.config.forced_decoder_ids
    #     elif (
    #         hasattr(self.generation_config, "forced_decoder_ids")
    #         and self.generation_config.forced_decoder_ids is not None
    #     ):
    #         forced_decoder_ids = self.generation_config.forced_decoder_ids
    #     else:
    #         forced_decoder_ids = kwargs.get("forced_decoder_ids", None)

    #     if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
    #         forced_decoder_ids = []
    #         if hasattr(generation_config, "language"):
    #             if generation_config.language in generation_config.lang_to_id.keys():
    #                 language_token = generation_config.language
    #             elif generation_config.language in TO_LANGUAGE_CODE.keys():
    #                 language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
    #             elif generation_config.language in TO_LANGUAGE_CODE.values():
    #                 language_token = f"<|{generation_config.language}|>"
    #             else:
    #                 is_language_code = len(generation_config.language) == 2
    #                 raise ValueError(
    #                     f"Unsupported language: {generation_config.language}. Language should be one of:"
    #                     f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
    #                 )
    #             forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
    #         else:
    #             forced_decoder_ids.append((1, None))  # automatically detect the language

    #         if hasattr(generation_config, "task"):
    #             if generation_config.task in TASK_IDS:
    #                 forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
    #             else:
    #                 raise ValueError(
    #                     f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
    #                 )
    #         elif hasattr(generation_config, "task_to_id"):
    #             forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
    #         if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
    #             idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
    #             forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

    #     if forced_decoder_ids is not None:
    #         generation_config.forced_decoder_ids = forced_decoder_ids

    #     if prompt_ids is not None:
    #         if kwargs.get("decoder_start_token_id") is not None:
    #             raise ValueError(
    #                 "When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten."
    #             )
    #         prompt_ids = prompt_ids.tolist()
    #         decoder_start_token_id, *text_prompt_ids = prompt_ids
    #         # Slicing the text prompt ids in a manner consistent with the OpenAI implementation
    #         # to accomodate context space for the prefix (see https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/decoding.py#L599)
    #         text_prompt_ids = text_prompt_ids[-self.config.max_target_positions // 2 - 1 :]
    #         # Set the decoder_start_token_id to <|startofprev|>
    #         kwargs.update({"decoder_start_token_id": decoder_start_token_id})

    #         # If the user passes `max_new_tokens`, increase its number to account for the prompt
    #         if kwargs.get("max_new_tokens", None) is not None:
    #             kwargs["max_new_tokens"] += len(text_prompt_ids)
    #             if kwargs["max_new_tokens"] >= self.config.max_target_positions:
    #                 raise ValueError(
    #                     f"The length of the sliced `prompt_ids` is {len(text_prompt_ids)}, and the `max_new_tokens` "
    #                     f"{kwargs['max_new_tokens'] - len(text_prompt_ids)}. Thus, the combined length of the sliced "
    #                     f"`prompt_ids` and `max_new_tokens` is: {kwargs['max_new_tokens']}. This exceeds the "
    #                     f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
    #                     "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
    #                     f"so that their combined length is less that {self.config.max_target_positions}."
    #                 )

    #         # Reformat the forced_decoder_ids to incorporate the prompt
    #         non_prompt_forced_decoder_ids = (
    #             kwargs.pop("forced_decoder_ids", None) or generation_config.forced_decoder_ids
    #         )
    #         forced_decoder_ids = [
    #             *text_prompt_ids,
    #             generation_config.decoder_start_token_id,
    #             *[token for _rank, token in non_prompt_forced_decoder_ids],
    #         ]
    #         forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
    #         generation_config.forced_decoder_ids = forced_decoder_ids

    #     if return_token_timestamps:
    #         kwargs["output_attentions"] = True
    #         return_dict_in_generate = True

    #         if getattr(generation_config, "task", None) == "translate":
    #             logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
    #         if not hasattr(generation_config, "alignment_heads"):
    #             raise ValueError(
    #                 "Model generation config has no `alignment_heads`, token-level timestamps not available. "
    #                 "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
    #             )

    #         if kwargs.get("num_frames") is not None:
    #             generation_config.num_frames = kwargs.pop("num_frames")

    #     if generation_config.return_timestamps is True:
    #         last_forced_decoder_ids = (
    #             generation_config.forced_decoder_ids[-1][-1]
    #             if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids
    #             else None
    #         )
    #         if last_forced_decoder_ids == self.generation_config.no_timestamps_token_id:
    #             # remove no_timestamp to be forcefully generated if we want to return timestamps
    #             # this is also important to make sure `WhisperTimeStampLogitsProcessor` functions correctly
    #             forced_decoder_ids = generation_config.forced_decoder_ids[:-1]
    #             # Make sure that if list is empty we set it to None
    #             generation_config.forced_decoder_ids = None if len(forced_decoder_ids) == 0 else forced_decoder_ids

    #         timestamp_processor = [WhisperTimeStampLogitsProcessor(generation_config)]
    #         logits_processor = (
    #             timestamp_processor if logits_processor is None else timestamp_processor + logits_processor
    #         )

    #     # 5. If we're in shortform mode, simple generate the whole input at once and return the output
    #     if is_shortform:
    #         outputs = super().generate(
    #             input_features,
    #             generation_config,
    #             logits_processor,
    #             stopping_criteria,
    #             prefix_allowed_tokens_fn,
    #             synced_gpus,
    #             return_dict_in_generate=return_dict_in_generate,
    #             **kwargs,
    #         )

    #         if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
    #             num_frames = getattr(generation_config, "num_frames", None)
    #             outputs["token_timestamps"] = self._extract_token_timestamps(
    #                 outputs, generation_config.alignment_heads, num_frames=num_frames
    #             )

    #         return outputs

    #     # 6. Else we're in longform mode which is more complex. We need to chunk the audio input depending on when the model generated
    #     # timestamp tokens
    #     # 6.1 Set running parameters for while loop
    #     if not return_segments and return_dict_in_generate:
    #         raise ValueError(
    #             "Make sure to set `return_segments=True` to return generation outputs as part of the `'segments' key.`"
    #         )

    #     # if input is longer than 30 seconds we default to long-form generation
    #     timestamp_begin = self.generation_config.no_timestamps_token_id + 1
    #     # input stride is mel frames per encoder output vector which is the product of all conv strides
    #     batch_size = input_features.shape[0]

    #     if batch_size > 1 and attention_mask is None:
    #         raise ValueError(
    #             "When doing long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` "
    #         )
    #     elif batch_size > 1:
    #         max_frames = attention_mask.sum(-1).cpu().to(torch.long)
    #         seek = torch.zeros((batch_size,), dtype=torch.long)
    #     else:
    #         max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
    #         seek = torch.zeros((1,), dtype=torch.long)

    #     current_segments = [[] for _ in range(batch_size)]
    #     cur_to_prev_index_map = list(range(batch_size))

    #     # batch size can decrease during the run
    #     cur_bsz = prev_bsz = batch_size

    #     # 6.2 Transcribe audio until we reach the end of all input audios
    #     while (seek < max_frames).any():
    #         prev_bsz = cur_bsz

    #         # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
    #         # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
    #         # to know which original audio is being decoded
    #         new_cur_to_prev_index_map = []
    #         for i in range(prev_bsz):
    #             prev_i = cur_to_prev_index_map[i]
    #             if seek[prev_i] >= max_frames[prev_i]:
    #                 cut_index = i + (cur_bsz - prev_bsz)
    #                 cur_bsz -= 1
    #                 input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1 :]], dim=0)
    #             else:
    #                 # cut out index that goes away
    #                 new_cur_to_prev_index_map.append(prev_i)

    #         # 6.4  Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
    #         cur_to_prev_index_map = new_cur_to_prev_index_map
    #         time_offset = seek * time_precision / input_stride
    #         seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

    #         # 6.5 Make sure that all inputs are padded to the same input length
    #         segment_input = []
    #         for i in range(cur_bsz):
    #             prev_i = cur_to_prev_index_map[i]
    #             segment_input_slice = input_features[
    #                 i : i + 1, :, seek[prev_i] : seek[prev_i] + seek_num_frames[prev_i]
    #             ]

    #             if segment_input_slice.shape[-1] < num_segment_frames:
    #                 # pad to 3000 if necessary
    #                 segment_input_slice = F.pad(
    #                     segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1])
    #                 )

    #             segment_input.append(segment_input_slice)

    #         segment_input = torch.cat(segment_input, dim=0)

    #         # 6.6 Batch generate current chunk
    #         seek_outputs = super().generate(
    #             segment_input,
    #             generation_config,
    #             logits_processor,
    #             stopping_criteria,
    #             prefix_allowed_tokens_fn,
    #             synced_gpus,
    #             return_dict_in_generate=return_dict_in_generate,
    #             **kwargs,
    #         )

    #         if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
    #             num_frames = getattr(generation_config, "num_frames", None)
    #             seek_outputs["token_timestamps"] = self._extract_token_timestamps(
    #                 seek_outputs, generation_config.alignment_heads, num_frames=num_frames
    #             )

    #         if return_dict_in_generate:
    #             seek_sequences = seek_outputs["sequences"]
    #             seek_outputs = [
    #                 {k: v[i] for k, v in seek_outputs.items()}
    #                 for i in range(next(iter(seek_outputs.values())).size(0))
    #             ]
    #         else:
    #             seek_sequences = seek_outputs

    #         # 6.7 Loop over each decoded audio individually as each decoding can be of a different length
    #         for i, seek_sequence in enumerate(seek_sequences):
    #             prev_i = cur_to_prev_index_map[i]

    #             # make sure we cut a predicted EOS token if we are not finished with the generation yet
    #             is_not_final = (seek[prev_i] + num_segment_frames) < max_frames[prev_i]
    #             if is_not_final and seek_sequence[-1] == self.generation_config.eos_token_id:
    #                 seek_sequence = seek_sequence[:-1]

    #             # remove all padding tokens
    #             if seek_sequence[-1] == self.generation_config.pad_token_id:
    #                 num_paddings = (seek_sequence == self.generation_config.pad_token_id).sum()
    #                 seek_sequence = seek_sequence[:-num_paddings]

    #             segments, segment_offset = self._retrieve_segment(
    #                 seek_sequence=seek_sequence,
    #                 seek_outputs=seek_outputs,
    #                 time_offset=time_offset,
    #                 timestamp_begin=timestamp_begin,
    #                 seek_num_frames=seek_num_frames,
    #                 cur_bsz=cur_bsz,
    #                 time_precision=time_precision,
    #                 input_stride=input_stride,
    #                 prev_idx=prev_i,
    #                 idx=i,
    #             )

    #             current_segments[prev_i] += segments
    #             seek[prev_i] += segment_offset

    #     # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
    #     # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
    #     sequences = []
    #     max_total_length = 0
    #     for current_segment_list in current_segments:
    #         sequences.append(torch.cat([d["tokens"] for d in current_segment_list], dim=-1))
    #         max_total_length = max(max_total_length, len(sequences[-1]))

    #     for i in range(batch_size):
    #         sequences[i] = F.pad(
    #             sequences[i], pad=(0, max_total_length - len(sequences[i])), value=self.generation_config.pad_token_id
    #         )

    #     sequences = torch.stack(sequences, dim=0)

    #     # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
    #     if return_segments:
    #         return {"sequences": sequences, "segments": current_segments}

    #     return sequences
    
    # def prepare_inputs_for_generation(
    #     self,
    #     decoder_input_ids,
    #     past_key_values=None,
    #     use_cache=None,
    #     encoder_outputs=None,
    #     attention_mask=None,
    #     **kwargs,
    # ):
    #     if past_key_values is not None:
    #         past_length = past_key_values[0][0].shape[2]

    #         # Some generation methods already pass only the last input ID
    #         if decoder_input_ids.shape[1] > past_length:
    #             remove_prefix_length = past_length
    #         else:
    #             # Default to old behavior: keep only final ID
    #             remove_prefix_length = decoder_input_ids.shape[1] - 1

    #         decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

    #     return {
    #         "encoder_outputs": encoder_outputs,
    #         "past_key_values": past_key_values,
    #         "decoder_input_ids": decoder_input_ids,
    #         "use_cache": use_cache,
    #         "decoder_attention_mask": None,
    #     }
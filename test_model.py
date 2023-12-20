#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple

import datasets
from datasets import interleave_datasets

import evaluate
import torch

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, CLIPProcessor
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from src.speech_img_text_paired_dataset import load_speech_img_text_paired_dataset, SpeechImgTextPairedDataCollator
from src.modeling_finetunewhisper import FinetuneWhisperModel
from src.configuration_finetunewhisper import FinetuneWhisperConfig


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    whisper_model: str = field(
        default="openai/whisper-small", metadata={"help": "the path of whisper model"}
    )
    
    language: str = field(
        default="english", metadata={"help": "the default language of whisper model"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data: str = field(
        metadata={
            "help": "the root to load dataset"
        },
    )

    manifest_files: str = field(
        default="",
        metadata={
            "help": "The name of the training unit text paired set split to use."
        },
    )


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    whisper_config = WhisperConfig.from_pretrained(model_args.whisper_model)
    finetune_whisper_config = FinetuneWhisperConfig(whisper_config.to_dict())
    # 4. Load tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(model_args.whisper_model)
    tokenizer.set_prefix_tokens(language=model_args.language)
    processor = WhisperProcessor.from_pretrained(model_args.whisper_model)
    extractor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_model)

    ### 5. Load dataset
    processor = CLIPProcessor.from_pretrained('/home/zth/work/new/Unconstrained-AVSR/pretrained_models/clip-vit-base-patch32')
    dataset = load_speech_img_text_paired_dataset(
        data_args.data,
        data_args.manifest_files,
        tokenizer,
        processor,
    )
    # 6. Load pretrained model
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently


    # model = FinetuneWhisperModel(finetune_whisper_config).from_pretrained('checkpoints/librispeech-100')
    # model = model.whisper_model
    model = FinetuneWhisperModel.from_pretrained('/home/zth/work/new/Unconstrained-AVSR/checkpoints/now_how2_2')
    # model = WhisperForConditionalGeneration.from_pretrained('pretrained_models/whisper-small')
    model.config.keys_to_ignore_at_inference = ["past_key_values"]
    if finetune_whisper_config.freeze_whisper:
        for name, param in model.whisper_model.named_parameters():
            param.requires_grad = False

    # 7. Define data collator
    data_collator = SpeechImgTextPairedDataCollator(
        tokenizer=tokenizer,
        pad_id=tokenizer.pad_token_id,
        sampling_rate=extractor.sampling_rate,
        extractor=extractor
    )
    
    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # compute orthographic wer
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
        # compute normalised WER
        pred_str_norm = [normalizer(pred) for pred in pred_str]
        label_str_norm = [normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str_norm = [
            pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i]
            for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]

        wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

        return {"wer_ortho": wer_ortho, "wer": wer}

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """

        pred_ids = torch.argmax(logits[1], dim=-1)
        return pred_ids
    
    # 7. Initialize Trainer
    tester = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8. Training
    output = tester.evaluate()
    print(output)

if __name__ == "__main__":
    main()
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers import WhisperConfig

from transformers.deepspeed import is_deepspeed_zero3_enabled

try:
    from .configuration_finetunewhisper import FinetuneWhisperConfig
    from .modeling_whisper import WhisperForConditionalGeneration
except:
    from src.configuration_finetunewhisper import FinetuneWhisperConfig
    from src.modeling_whisper import WhisperForConditionalGeneration
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize


from transformers import CLIPProcessor, CLIPModel

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

class Adapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
    ):
        super(Adapter, self).__init__()

        self.activation = nn.GELU()
        self.fc1 = nn.Linear(in_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, out_dim, bias=False)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class FinetuneWhisperModel(PreTrainedModel):
    config_class = FinetuneWhisperConfig
    base_model_prefix = "finewhisper"

    def __init__(self, config: FinetuneWhisperConfig):
        super().__init__(config)
        
        # clip model
        self.clip_model = CLIPModel.from_pretrained('pretrained_models/clip-vit-base-patch32')  # clip.available_models()
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.whisper_config.if_cross_attn = config.if_cross_attn
        self.whisper_config._attn_implementation = "default" # eager: Attention, flash_attention_2: flash attention, sdpa: sdpa attention (need update transformers)
        self.whisper_model = WhisperForConditionalGeneration(self.whisper_config)
        model_state_dict = torch.load('/home/zth/work/new/Unconstrained-AVSR/pretrained_models/whisper-small/pytorch_model.bin')
        self.whisper_model.load_state_dict(model_state_dict, strict=False)
        
        self.clip_adapter = Adapter(self.clip_model.config.projection_dim, config.adapter_inner_dim, self.whisper_config.d_model)

    def forward(self, image_feature, labels, input_features, attention_mask):
        
        image_features = self.clip_model.get_image_features(image_feature)
        image_features = self.clip_adapter(image_features)

        return self.whisper_model(image_features, labels=labels, input_features=input_features, attention_mask=attention_mask)

    def generate(self, **kwargs):
        
        image_feature = kwargs.pop('image_feature')
        image_features = self.clip_model.get_image_features(image_feature)
        image_features = self.clip_adapter(image_features)
        kwargs['image_features'] = image_features
        return self.whisper_model.generate(**kwargs)
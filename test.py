import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple

import datasets
from datasets import interleave_datasets
from torch.utils.data import DataLoader
import logging

import evaluate
import torch
import numpy as np
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

from src.speech_img_text_paired_dataset import (load_speech_img_text_paired_dataset, 
        SpeechImgTextPairedDataCollator,
        collate_tokens,
        get_waveform)
from src.modeling_finetunewhisper import FinetuneWhisperModel
from src.configuration_finetunewhisper import FinetuneWhisperConfig

from transformers import GenerationConfig
from tqdm import tqdm
def setup_logger(log_file):
    # 创建一个logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 创建一个文件handler，用于写入日志文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # 创建一个控制台handler，用于输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将handler添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


log_file = 'log.txt'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = 'checkpoints/now_how2_3'
data_path = 'data/now_how2_val'
batch_size=4


# 调用setup_logger函数创建logger
logger = setup_logger(log_file)

generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=False,
    temperature=0.9,
    top_p=0.0,
    num_beams=1,
    num_return_sequences=1,
)

def main():
    tokenizer = WhisperTokenizer.from_pretrained('pretrained_models/whisper-small')
    generation_config.update(
        **{
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    )
    extractor = WhisperFeatureExtractor.from_pretrained('pretrained_models/whisper-small')
    processor = CLIPProcessor.from_pretrained('/home/zth/work/new/Unconstrained-AVSR/pretrained_models/clip-vit-base-patch32')
    dataset = load_speech_img_text_paired_dataset(
        dataroot=data_path,
        manifest_files="*.jsonl",
        tokenizer=tokenizer,
        processor=processor,
    )
    model = FinetuneWhisperModel.from_pretrained(model_path)
    model.eval()
    model.to(device)
 
    
    def _collate_fn(samples):
        input_ids_batch = [{"input_ids": sample["input_ids"]} for sample in samples]
        labels_batch = tokenizer.pad(input_ids_batch, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        raw_speech = [
            get_waveform(sample["audio_path"], output_sample_rate=16000) for sample in samples
        ]
        speech_inputs = extractor(
            raw_speech, 
            sampling_rate=16000, 
            return_attention_mask=True,
            return_tensors="pt"
        )

        img_features_batch = [sample['image_feature']['pixel_values'] for sample in samples]
        img_features_batch = np.array(img_features_batch)
        img_features = torch.from_numpy(img_features_batch).squeeze(1)  # numpy 转 torch.Tensor
        
        audio_path = [sample['audio_path'] for sample in samples]
        
        return { 
            "audio_path": audio_path,
            "image_feature": img_features,
            "labels": labels,
            "input_features": speech_inputs.input_features,
            "attention_mask": speech_inputs.attention_mask,
        }
        
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)
    
    # 7. Initialize Trainer
    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()
    
    # def compute_metrics(pred):
    #     import pdb
    #     pdb.set_trace()
    #     pred_ids = pred.predictions
    #     label_ids = pred.label_ids

    #     # replace -100 with the pad_token_id
    #     label_ids[label_ids == -100] = tokenizer.pad_token_id

    #     # we do not want to group tokens when computing the metrics
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    #     # compute orthographic wer
    #     wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
    #     # compute normalised WER
    #     pred_str_norm = [normalizer(pred) for pred in pred_str]
    #     label_str_norm = [normalizer(label) for label in label_str]
    #     # filtering step to only evaluate the samples that correspond to non-zero references:
    #     pred_str_norm = [
    #         pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    #     ]
    #     label_str_norm = [
    #         label_str_norm[i]
    #         for i in range(len(label_str_norm))
    #         if len(label_str_norm[i]) > 0
    #     ]

    #     wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    #     return {"wer_ortho": wer_ortho, "wer": wer}

    for batch in tqdm(eval_dataloader):
        
        audio_paths = batch.pop('audio_path')

        batch = {k: v.to(device) for k, v in batch.items()}
                
        with torch.no_grad():
            outputs = model.generate(generation_config=generation_config, **batch)

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        for prediction, reference, audio_path in zip(predictions, references, audio_paths):
            logger.info("audio_path: " + audio_path)
            logger.info("hyp: " + prediction)
            logger.info("ref: " + reference + '\n')
        metric.add_batch(predictions=predictions, references=references)
    resutls = metric.compute()
    logger.info('metric result: {}'.format(resutls))
if __name__ == "__main__":
    main()
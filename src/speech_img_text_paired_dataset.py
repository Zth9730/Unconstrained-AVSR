import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire
import soundfile as sf
import io
from PIL import Image
import numpy as np
import torch
import random
import datasets
from dataclasses import dataclass
import subprocess
from transformers import WhisperTokenizer, WhisperFeatureExtractor
import av
from transformers import CLIPProcessor

logger = logging.getLogger(__name__)

def process_dataset(batch, tokenizer, processor):
    audio_path = batch["audio"]
    try:
        info = sf.info(audio_path)
        input_length = info.duration
    except:
        input_length = 0.0
        
    imgs =  Image.open(batch["img"])
    img_inputs = processor(images=imgs, return_tensors="pt")
    text_features = tokenizer(batch["text"])
    batch["input_ids"] = text_features['input_ids']
    batch["attention_mask"] = text_features['attention_mask']
    batch["audio_path"] = audio_path
    batch["input_length"] = input_length
    batch["image_feature"] = img_inputs
    batch['audio_path'] = audio_path

    return batch

def load_speech_img_text_paired_dataset(
    dataroot="",
    manifest_files="",
    tokenizer=None,
    processor=None,
    num_proc=8,
    sort=False,
    shuffle=True,
    seed=42,
    max_label_length = 225,
    min_input_length = 0.0,
    max_input_length = 30.0,
    num_img=2,
):
    if os.path.exists(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all"))):
        logger.warning("load processed dataset")
        dataset = datasets.load_from_disk(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all")))
        if shuffle:
            dataset = dataset.shuffle(seed)
        return dataset
    
    logger.warning(f"load dataset from scratch from {dataroot}/{manifest_files}")

    manifest_files_list = manifest_files.split(",")
    raw_dataset = datasets.load_dataset(
        dataroot, data_files=manifest_files_list, split="train", streaming=False
    )

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "processor": processor,
        },
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        num_proc=num_proc,
    )

    def is_in_length_range(length, labels):
        return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length
    
    dataset = dataset.filter(
        is_in_length_range,
        input_columns=["input_length", "input_ids"],
    )

    dataset.save_to_disk(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all")))

    return dataset


def collate_tokens(
        values: List[List[int]],
        pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][: len(v)])

    return res

def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
    meta = path_or_fp.split(":")
    if len(meta) == 3 and (meta[0].endswith(".wav") or meta[0].endswith(".flac")):
        path_or_fp = meta[0]
        start = int(meta[1])
        frames = int(meta[2])
    else:
        path_or_fp = path_or_fp


    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext in [".wav", ".flac", ".ogg", ".mp3"]:
            pass
        else:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLACC/OGG/MP3 audios")

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T

    waveform, sample_rate = convert_waveform(waveform, sample_rate, to_mono=mono, to_sample_rate=output_sample_rate)
    if not normalization:
        waveform *= 2 ** 15
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform

def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


@dataclass
class SpeechImgTextPairedDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    tokenizer: WhisperTokenizer
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()

    def __call__(self, samples: List[Dict]):
        
        input_ids_batch = [{"input_ids": sample["input_ids"]} for sample in samples]
        labels_batch = self.tokenizer.pad(input_ids_batch, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        raw_speech = [
            get_waveform(sample["audio_path"], output_sample_rate=self.sampling_rate) for sample in samples
        ]
        speech_inputs = self.extractor(
            raw_speech, 
            sampling_rate=self.sampling_rate, 
            return_attention_mask=True,
            return_tensors="pt"
        )

        img_features_batch = [sample['image_feature']['pixel_values'] for sample in samples]
        img_features_batch = np.array(img_features_batch)
        img_features = torch.from_numpy(img_features_batch).squeeze(1)  # numpy 转 torch.Tensor

        
        return { 
            "image_feature": img_features,
            "labels": labels,
            "input_features": speech_inputs.input_features,
            "attention_mask": speech_inputs.attention_mask,
        }


def offline_process(
    dataroot="",
    manifest_files="",
    whisper_path="",
    clip_path="",
    num_proc=8,
):
    tokenizer = WhisperTokenizer.from_pretrained(whisper_path)
    processor = CLIPProcessor.from_pretrained(clip_path)
    dataset = load_speech_img_text_paired_dataset(
        dataroot,
        manifest_files,
        tokenizer,
        processor,
        num_proc
    )
    for key in dataset[0].keys():
        if key != "audio_path" and key != "is_readable" and key != "input_length":
            print(key, len(dataset[0][key]))
        else:
            print(key, dataset[0][key])
    print(len(dataset))


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })
# src/data.py

import os
from typing import Optional, Dict, Any, List
from PIL import Image
import torch
from datasets import Dataset

def load_caption_dataset(csv_path: str, image_col: str, text_col: str, style_col: Optional[str]) -> Dataset:
    # Rely on datasets to load the CSV; image paths are in the CSV already
    from datasets import load_dataset
    ds = load_dataset("csv", data_files=csv_path)["train"]
    # make sure columns exist
    assert image_col in ds.column_names and text_col in ds.column_names, \
        f"CSV must have columns: {image_col}, {text_col}"
    if style_col and style_col not in ds.column_names:
        style_col = None
    return ds

def _open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def preprocess_examples(
    batch: Dict[str, List[Any]],
    processor,
    prompt_template: str,
    image_col: str,
    text_col: str,
    style_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Expects batched inputs (lists). Returns lists, one item per example, for each field.
    """
    # 1) Load images
    img_paths: List[str] = batch[image_col]
    images: List[Image.Image] = [_open_rgb(p) for p in img_paths]

    # 2) Build prompts (inputs) and targets (labels)
    captions: List[str] = batch[text_col]
    styles: List[Optional[str]] = batch[style_col] if (style_col and style_col in batch) else [None] * len(captions)

    prompts: List[str] = []
    for cap, sty in zip(captions, styles):
        prompt = prompt_template.strip()
        if sty and str(sty).strip():
            prompt += f"\nStyle: {sty}"
        prompt += "\nCaption:"
        prompts.append(prompt)

    # 3) Encode inputs with the processor
    model_inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # 4) Labels are the ground-truth captions (target text)
    labels = processor.tokenizer(
        captions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    model_inputs["labels"] = labels

    # 5) Convert batch tensors to lists (one item per example) so 🤗 Datasets can write them
    out: Dict[str, Any] = {}
    batch_size = len(images)

    def split_to_lists(t: torch.Tensor) -> List:
        # Split along batch dim into per-example arrays/lists
        if t.dim() == 0:
            # scalar -> repeat to match batch or wrap
            return [t.item()] * batch_size
        if t.size(0) != batch_size:
            # broadcasted or single tensor: just convert entire thing
            return t.detach().cpu().numpy().tolist()
        # proper batched tensor
        chunks = t.detach().cpu().split(1, dim=0)
        items = [c.squeeze(0).numpy() for c in chunks]
        return items

    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = split_to_lists(v)
        else:
            out[k] = v  # rare, usually tensors

    return out
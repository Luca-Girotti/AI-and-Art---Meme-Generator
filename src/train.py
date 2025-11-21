import os
import yaml
import random
import argparse
import time
import datetime
import re  # <- for regex caption cleaning
import torch
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image

from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    InstructBlipForConditionalGeneration,
    get_linear_schedule_with_warmup,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from tqdm.auto import tqdm

from .data import load_caption_dataset  # CSV loader


@dataclass
class TrainConfig:
    run_name: str
    seed: int
    model_name: str
    quantization: Optional[str]
    train_csv: str
    val_csv: Optional[str]
    image_col: str
    text_col: str
    style_col: Optional[str]
    prompt_template: str
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list
    max_steps: Optional[int]
    save_steps: int
    eval_steps: int
    logging_steps: int
    num_beams: int
    max_new_tokens: int
    fp16: bool
    bf16: bool


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


ALLOW_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "labels",
    "qformer_input_ids",
    "qformer_attention_mask",
}


def clean_keys(batch: dict) -> dict:
    return {k: v for k, v in batch.items() if (k in ALLOW_KEYS and "embeds" not in k)}


def freeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def _fix_batch_dims(batch: Dict[str, Any]) -> Dict[str, Any]:
    def _squeeze_middle(t: torch.Tensor):
        if t.dim() == 5 and t.size(1) == 1:
            return t.squeeze(1)
        if t.dim() == 3 and t.size(1) == 1:
            return t.squeeze(1)
        return t

    for key in ("pixel_values", "input_ids", "attention_mask", "qformer_input_ids", "qformer_attention_mask", "labels"):
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = _squeeze_middle(batch[key])
        elif key in batch and isinstance(batch[key], list):
            try:
                t = torch.tensor(batch[key])
                batch[key] = _squeeze_middle(t)
            except Exception:
                pass
    return batch


def _save_lora_only(
    accelerator: Accelerator,
    model: torch.nn.Module,
    processor,
    base_dir: str,
    tag: Optional[str] = None,
):
    lora_dir = os.path.join(base_dir, f"lora_adapter{'' if not tag else f'_{tag}'}")
    os.makedirs(lora_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    # adapter lives inside language_model
    unwrapped.language_model.save_pretrained(lora_dir)
    processor.save_pretrained(base_dir)
    accelerator.print(f"[save] LoRA adapter saved to: {lora_dir}")


def main(cfg: TrainConfig):
    torch.set_num_threads(min(4, os.cpu_count() or 4))  # allow light multi-threading

    use_cpu = os.getenv("ACCELERATE_USE_CPU", "0").lower() in ("1", "true", "yes")
    accelerator = Accelerator(
        cpu=use_cpu,
        mixed_precision="no" if use_cpu else ("fp16" if cfg.fp16 else ("bf16" if cfg.bf16 else "no")),
    )
    device = accelerator.device
    accelerator.print(f"[debug] using device: {device}, cpu={use_cpu}")

    set_seed(cfg.seed)

    processor = AutoProcessor.from_pretrained(cfg.model_name)

    # === Load data ===
    train_ds = load_caption_dataset(cfg.train_csv, cfg.image_col, cfg.text_col, cfg.style_col)
    if cfg.val_csv:
        val_ds = load_caption_dataset(cfg.val_csv, cfg.image_col, cfg.text_col, cfg.style_col)
    else:
        split = train_ds.train_test_split(test_size=0.05, seed=cfg.seed)
        train_ds, val_ds = split["train"], split["test"]

    # --- Cap dataset sizes ---
    TRAIN_CAP = 700
    VAL_CAP = 200
    train_ds = train_ds.select(range(min(TRAIN_CAP, len(train_ds))))
    val_ds = val_ds.select(range(min(VAL_CAP, len(val_ds))))
    accelerator.print(f"[debug] capped dataset: train={len(train_ds)} val={len(val_ds)}")

    if os.getenv("DEBUG_SUBSET", "0").lower() in ("1", "true", "yes"):
        train_ds = train_ds.select(range(min(300, len(train_ds))))
        val_ds = val_ds.select(range(min(50, len(val_ds))))
        accelerator.print(f"[debug] DEBUG_SUBSET active: train={len(train_ds)} val={len(val_ds)}")

    # === Helper: reduce MEMECAP-style text to a short meme caption ===
    def _to_meme_caption(text: str) -> str:
        """
        Turn long MEMECAP-style explanations into a short caption, e.g.:

        - Use `title "..."/title: ...` if present
        - Use the text after `Answer:` if present
        - Combine as: `title – answer`
        - Strip boilerplate and hard-cap to ~20 words
        """
        if not isinstance(text, str):
            text = str(text)

        # Collapse whitespace/newlines
        clean = re.sub(r"\s+", " ", text).strip()

        # Try to extract a title
        title = ""
        m = re.search(r'[Tt]itle\s*["“](.+?)["”]', clean)
        if not m:
            m = re.search(r'[Tt]itle\s*[:\-]\s*"?([^"”]+?)"?([\.!?]|$)', clean)
        if m:
            title = m.group(1).strip()

        # Try to extract the "Answer: ..." bit
        answer = ""
        m = re.search(r'[Aa]nswer\s*[:\-]\s*(.+?)(?:[Ee]xplanation[:\-]|[Cc]ontext[:\-]|$)', clean)
        if m:
            answer = m.group(1).strip()

        pieces = []
        if title:
            pieces.append(title)
        if answer:
            pieces.append(answer)

        if pieces:
            caption = " – ".join(pieces)
        else:
            # Fallback: use whole thing but keep it short
            caption = clean

        # Remove some obvious boilerplate intros
        caption = re.sub(r'^(meme (poster|image)\s+is\s+)', "", caption, flags=re.IGNORECASE).strip()
        caption = re.sub(r'^(this (meme|image)\s+shows\s+)', "", caption, flags=re.IGNORECASE).strip()

        # Hard cap to 20 words
        words = caption.split()
        if len(words) > 20:
            caption = " ".join(words[:20])

        return caption.strip()

    # === Preprocess ===
    def _preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        img_path = batch[cfg.image_col][0]
        raw_caption = batch[cfg.text_col][0]
        caption = _to_meme_caption(raw_caption)

        style = None
        if cfg.style_col and cfg.style_col in batch and batch[cfg.style_col]:
            style = batch[cfg.style_col][0]

        img = Image.open(img_path).convert("RGB")

        prompt = cfg.prompt_template.strip()
        if style and str(style).strip():
            prompt += f"\nStyle: {style}"
        prompt += "\nCaption:"

        enc = processor(images=img, text=prompt, return_tensors="pt")

        input_ids = enc["input_ids"].squeeze(0).tolist()
        attention_mask = enc["attention_mask"].squeeze(0).tolist()
        pixel_values = enc["pixel_values"].squeeze(0).tolist()

        qformer_input_ids = enc.get("qformer_input_ids", None)
        qformer_attention = enc.get("qformer_attention_mask", None)
        if qformer_input_ids is not None:
            qformer_input_ids = qformer_input_ids.squeeze(0).tolist()
        if qformer_attention is not None:
            qformer_attention = qformer_attention.squeeze(0).tolist()

        labels = processor.tokenizer(
            caption,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).input_ids.squeeze(0).tolist()

        out = {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
            "pixel_values": [pixel_values],
            "labels": [labels],
        }
        if qformer_input_ids is not None:
            out["qformer_input_ids"] = [qformer_input_ids]
        if qformer_attention is not None:
            out["qformer_attention_mask"] = [qformer_attention]

        return out

    train_ds = train_ds.map(_preprocess, batched=True, batch_size=1, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(_preprocess, batched=True, batch_size=1, remove_columns=val_ds.column_names)

    train_loader_plain = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=0,
    )
    val_loader_plain = DataLoader(
        val_ds,
        batch_size=cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=0,
    )

    # === Model setup ===
    load_8bit = cfg.quantization == "8bit"
    load_4bit = cfg.quantization == "4bit"
    model = InstructBlipForConditionalGeneration.from_pretrained(
        cfg.model_name,
        load_in_8bit=load_8bit,
        load_in_4bit=load_4bit,
        device_map="auto" if (load_8bit or load_4bit) else None,
    )
    if load_8bit or load_4bit:
        model = prepare_model_for_kbit_training(model)

    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "vision_model"):
        freeze_module(model.vision_model)
    if hasattr(model, "qformer"):
        freeze_module(model.qformer)

    if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "embed_tokens"):
        model.language_model.model.embed_tokens.weight.requires_grad = False
    if hasattr(model.language_model, "lm_head"):
        model.language_model.lm_head.weight.requires_grad = False

    lm = model.language_model
    task_type = "SEQ_2_SEQ_LM" if lm.__class__.__name__.lower().startswith("t5") else "CAUSAL_LM"

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=task_type,
    )
    lm = get_peft_model(lm, peft_config)
    model.language_model = lm

    accelerator.print("[debug] starting warmup forward on 1 batch (pre-prepare)…")
    warm_loader = DataLoader(
        train_ds.select(range(1)),
        batch_size=1,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=0,
    )
    warm_batch = next(iter(warm_loader))
    warm_batch = clean_keys(warm_batch)
    warm_batch = _fix_batch_dims(warm_batch)
    t0 = time.time()
    with torch.no_grad():
        _ = model(**warm_batch)
    accelerator.print(f"[debug] warmup forward OK in {time.time()-t0:.2f}s")

    model, train_loader, val_loader = accelerator.prepare(model, train_loader_plain, val_loader_plain)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_train_epochs // max(cfg.gradient_accumulation_steps, 1)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    warmup_steps = int(total_steps * cfg.warmup_ratio) if total_steps > 0 else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    printed_keys = False

    # === TRAIN LOOP ===
    for epoch in range(cfg.num_train_epochs):
        if accelerator.is_main_process:
            accelerator.print(f"\n===== Epoch {epoch+1}/{cfg.num_train_epochs} =====")

        model.train()
        progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        epoch_loss = 0.0
        step_start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = clean_keys(batch)
            batch = _fix_batch_dims(batch)

            if not printed_keys and accelerator.is_main_process:
                accelerator.print(f"[debug] batch keys: {list(batch.keys())}")
                printed_keys = True

            outputs = model(**batch)
            loss = outputs.loss / max(cfg.gradient_accumulation_steps, 1)
            accelerator.backward(loss)
            epoch_loss += float(loss.detach())

            if accelerator.is_main_process:
                avg_loss = epoch_loss / step
                elapsed = time.time() - step_start
                steps_left = len(train_loader) - step
                eta = datetime.timedelta(seconds=int((elapsed / step) * steps_left))
                progress.set_postfix(loss=f"{avg_loss:.4f}", eta=str(eta))
                progress.update(1)

            if (step % cfg.gradient_accumulation_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if cfg.logging_steps and global_step % cfg.logging_steps == 0 and accelerator.is_main_process:
                    accelerator.print(f"epoch {epoch} step {global_step} loss {outputs.loss.item():.4f}")

            if cfg.max_steps and global_step >= cfg.max_steps:
                break

        progress.close()

        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / max(1, len(train_loader))
            accelerator.print(f"Epoch {epoch+1} finished – avg loss {avg_epoch_loss:.4f}")

        if cfg.max_steps and global_step >= cfg.max_steps:
            break

    if accelerator.is_main_process:
        _save_lora_only(accelerator, model, processor, save_dir, tag="final")


def evaluate(model, data_loader, accelerator: Accelerator):
    model.eval()
    total, count = 0.0, 0
    progress = tqdm(total=len(data_loader), desc="Eval", disable=not accelerator.is_main_process, leave=False)
    with torch.no_grad():
        for batch in data_loader:
            batch = clean_keys(batch)
            batch = _fix_batch_dims(batch)
            outputs = model(**batch)
            total += outputs.loss.item()
            count += 1
            progress.update(1)
    progress.close()
    model.train()
    return total / max(count, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    for k in ["learning_rate", "weight_decay", "warmup_ratio"]:
        if k in raw and isinstance(raw[k], str):
            raw[k] = float(raw[k])
    for k in ["per_device_train_batch_size", "per_device_eval_batch_size",
              "gradient_accumulation_steps", "num_train_epochs",
              "save_steps", "eval_steps", "logging_steps"]:
        if k in raw and isinstance(raw[k], str):
            raw[k] = int(raw[k])
    cfg = TrainConfig(**raw)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
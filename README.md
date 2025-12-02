# Meme Generator — InstructBLIP (LoRA)

End‑to‑end project to fine‑tune an open‑source **InstructBLIP (flan-t5-xl)** model for witty, sarcastic meme captions given an image, and to serve a Gradio UI that overlays the caption on top of the image.

## Why InstructBLIP?
- It’s a visual‑language instruction‑tuned model with strong image captioning ability.
- The **flan-t5-xl** text head avoids Vicuna licensing complexity.
- We fine‑tune with **LoRA** (PEFT) for efficiency, optionally with 8‑bit/4‑bit loading.

## Project Structure
```
meme-generator-instructblip/
  configs/config.yaml
  dataset/sample.csv
  src/
    data.py
    train.py
    infer.py
    overlay.py
    ui_app.py
  scripts/
    prepare_dataset.py
  requirements.txt
  README.md
```

## Dataset format
Prepare a CSV (UTF-8) with columns:
- `image_path`: local path to the image file
- `caption`: target meme caption text (keep short, punchy)
- (optional) `style`: free text tag (e.g., "sarcastic", "wholesome", "ironic")

Example (see `dataset/sample.csv`). You can reuse assets or preprocessing ideas from your previous project; just export to this CSV schema.

## Quickstart

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure
Edit `configs/config.yaml` to point to your dataset and tweak training params.

### 3) Train (LoRA fine‑tuning)
```bash
python -m src.train --config configs/config.yaml
```
This will create a LoRA adapter under `outputs/<run_name>`.

### 4) Inference (CLI)
```bash
python -m src.infer --image path/to/image.jpg --adapter outputs/<run_name> --num_beams 5 --num_return_sequences 3
```

### 5) Gradio UI
```bash
python -m src.ui_app --adapter outputs/<run_name>
```
Upload an image, get several captions, and optionally overlay text on the image, then download.

## Notes
- Training uses **Accelerate** + **PEFT (LoRA)** and can do 8‑bit/4‑bit loading via **bitsandbytes** (optional).
- Model: `Salesforce/instructblip-flan-t5-xl` (image encoder + T5 text decoder).
- Prompts are styled for meme sarcasm; adjust in `data.py` / `infer.py` to fit your flavor.
- If you want to start from your older BLIP/GPT‑2 pipeline, adapt the CSV export and reuse any cleaning/normalization utilities.

### Use MemeCap + Memotion (Hugging Face)

You can build a single CSV from one or both datasets with:

```bash
# Example IDs — replace with the exact HF IDs you choose
python -m scripts.build_from_hf --hf_id linhduong/memecap --out dataset/memecap.csv
python -m scripts.build_from_hf --hf_id harishsng/memotion --out dataset/memotion.csv

# Or combine into one unified CSV
python -m scripts.build_from_hf --hf_id linhduong/memecap --hf_id harishsng/memotion --out dataset/memes_combo.csv
```

If auto-detection of columns fails, specify them explicitly:

```bash
python -m scripts.build_from_hf   --hf_id linhduong/memecap   --image_col image --caption_col caption --style_col humor   --out dataset/memecap.csv
```

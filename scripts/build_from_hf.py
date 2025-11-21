"""
Build a unified CSV from one or more Hugging Face datasets (e.g., MemeCap, Memotion).

This version supports schemas like:
  features: ['messages', 'images']
  - images: list[Image]
  - messages: list[{role:str, content:str}]  # caption appears as "Caption: ..."

It:
- Auto-detects columns (prioritizes 'images' and 'messages')
- Extracts "Caption: ..." from messages (works for both MemeCap & Memotion mirrors you showed)
- Re-encodes images to JPEG and skips corrupt rows
- Supports --max_items for quick prototyping
"""

import os
import re
import csv
import json
import shutil
import argparse
from typing import Optional, Tuple, Any

from datasets import load_dataset, Image, IterableDataset
from PIL import Image as PILImage
from PIL import ImageFile

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _extract_caption_from_messages(field: Any) -> str:
    """
    messages: list of dicts like {'role': 'user'/'assistant', 'content': '...'}
    We look for a line like "Caption: ...."
    If not found, fall back to concatenating contents.
    """
    try:
        if isinstance(field, list) and field:
            # 1) try explicit "Caption: ..."
            for m in field:
                if isinstance(m, dict) and "content" in m:
                    txt = str(m["content"])
                    mcap = re.search(r"Caption:\s*(.*)", txt, flags=re.IGNORECASE)
                    if mcap:
                        return mcap.group(1).strip().strip('"')
            # 2) fallback: join all contents
            joined = " ".join(str(m.get("content", "")) for m in field if isinstance(m, dict))
            return joined.strip()
        # Sometimes serialized as JSON string
        if isinstance(field, str):
            try:
                arr = json.loads(field)
                return _extract_caption_from_messages(arr)
            except Exception:
                return field.strip()
    except Exception:
        pass
    return ""


def _first_pil_from_images(images_field: Any):
    """
    images_field can be:
      - list of PIL Images / dicts with 'image'
      - single PIL Image / dict with 'image'
    Return first PIL.Image or None.
    """
    def to_pil(x):
        if isinstance(x, dict) and "image" in x:
            return x["image"]
        return x

    if isinstance(images_field, list):
        for it in images_field:
            pil = to_pil(it)
            if hasattr(pil, "save"):
                return pil
        return None
    else:
        pil = to_pil(images_field)
        return pil if hasattr(pil, "save") else None


def detect_columns(ds, image_col: Optional[str] = None,
                   caption_col: Optional[str] = None,
                   style_col: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    feats = ds.features

    # Image column (prefer 'images' then Image feature then commons)
    if image_col and image_col in feats:
        img_col = image_col
    else:
        img_col = "images" if "images" in feats else None
        if img_col is None:
            for k, v in feats.items():
                if isinstance(v, Image):
                    img_col = k
                    break
        if img_col is None:
            for k in ["image", "img", "filepath", "path", "file", "image_path"]:
                if k in feats:
                    img_col = k
                    break

    # Caption column (prefer 'messages' then obvious text fields)
    if caption_col and caption_col in feats:
        cap_col = caption_col
    else:
        cap_col = "messages" if "messages" in feats else None
        if cap_col is None:
            for k in ["caption", "text", "meme_text", "humor_caption", "funny_caption", "final_caption"]:
                if k in feats:
                    cap_col = k
                    break
        if cap_col is None:
            for k, v in feats.items():
                if "string" in str(v).lower():
                    cap_col = k
                    break

    # Optional style column
    sty_col = style_col if (style_col and style_col in feats) else None
    return img_col, cap_col, sty_col


def write_partition(ds,
                    img_col: str,
                    cap_col: str,
                    sty_col: Optional[str],
                    out_writer: csv.DictWriter,
                    tmpdir: str,
                    max_items: Optional[int] = None) -> int:
    saved = 0
    count = 0
    for ex in ds:
        if max_items is not None and count >= max_items:
            break
        count += 1

        try:
            # --- image ---
            pil = _first_pil_from_images(ex.get(img_col))
            if pil is None:
                continue
            out_path = os.path.join(tmpdir, f"img_{saved:08d}.jpg")
            try:
                pil.convert("RGB").save(out_path, format="JPEG", quality=90, optimize=True)
            except Exception:
                continue

            # --- caption ---
            raw_cap = ex.get(cap_col)
            if cap_col == "messages":
                caption = _extract_caption_from_messages(raw_cap)
            else:
                caption = ("" if raw_cap is None else str(raw_cap)).strip()
            if not caption:
                # Skip rows with no text
                continue

            # --- style (optional) ---
            style = ""
            if sty_col:
                raw_style = ex.get(sty_col)
                style = ("" if raw_style is None else str(raw_style)).strip()

            out_writer.writerow({"image_path": out_path, "caption": caption, "style": style})
            saved += 1

        except Exception:
            # Skip on any per-row failure
            continue

    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", action="append", required=True, help="Hugging Face dataset id (can pass multiple)")
    ap.add_argument("--split", default="train", help="Split name (default: train)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--image_col", default=None, help="Force image column name")
    ap.add_argument("--caption_col", default=None, help="Force caption column name")
    ap.add_argument("--style_col", default=None, help="Optional style column name")
    ap.add_argument("--max_items", type=int, default=None, help="Limit total items per dataset (debugging)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmproot = os.path.join(os.path.dirname(args.out), "_imgs_cache")
    if os.path.isdir(tmproot):
        shutil.rmtree(tmproot)
    os.makedirs(tmproot, exist_ok=True)

    total = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption", "style"])
        writer.writeheader()

        for hf_id in args.hf_id:
            print(f"[INFO] Loading {hf_id} ({args.split})")
            ds = load_dataset(hf_id, split=args.split)

            img_col, cap_col, sty_col = detect_columns(ds, args.image_col, args.caption_col, args.style_col)
            if not img_col or not cap_col:
                print(f"[WARN] Could not detect columns for {hf_id}: {list(ds.features.keys())}")
                continue

            tmpdir = os.path.join(tmproot, hf_id.replace("/", "_"))
            os.makedirs(tmpdir, exist_ok=True)

            if isinstance(ds, IterableDataset) and args.max_items:
                ds = ds.take(args.max_items)

            saved = write_partition(
                ds, img_col, cap_col, sty_col, writer, tmpdir, max_items=args.max_items
            )
            total += saved
            print(f"[INFO] Saved {saved} rows from {hf_id}")

    print(f"[DONE] Wrote {total} rows to {args.out}")


if __name__ == "__main__":
    main()
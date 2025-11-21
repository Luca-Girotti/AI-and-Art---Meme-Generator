"""
Helper to build a CSV from a folder of (image, caption) pairs.

Expected structure:
root/
  img001.jpg
  img001.txt   # contains the caption
  img002.png
  img002.txt
...

Usage:
python -m scripts.prepare_dataset --root path/to/folder --out dataset/mydata.csv
"""
import os, csv, argparse, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    rows = []
    for img in glob.glob(os.path.join(args.root, "*")):
        if img.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
            base, _ = os.path.splitext(img)
            txt = base + ".txt"
            if os.path.isfile(txt):
                with open(txt, "r", encoding="utf-8") as f:
                    cap = f.read().strip()
                rows.append({"image_path": img, "caption": cap, "style": ""})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path","caption","style"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} examples to {args.out}")

if __name__ == "__main__":
    main()

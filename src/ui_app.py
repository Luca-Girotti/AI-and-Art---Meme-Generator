import argparse
import os

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from .infer import generate as generate_captions


# --------------------------------------------------
# TEXT MEASUREMENT HELPER (Pillow >= 10 friendly)
# --------------------------------------------------
def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    """
    Measure text width using textbbox (works in newer Pillow versions).
    """
    if not text:
        return 0
    # textbbox returns (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


# --------------------------------------------------
# SIMPLE MEME RENDERER (overlay text on image)
# --------------------------------------------------
def _wrap_text(draw, text, font, max_width):
    """
    Wrap text so each line fits max_width.
    Uses textbbox() instead of deprecated textsize().
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for w in words[1:]:
        test = current + " " + w
        bbox = draw.textbbox((0, 0), test, font=font)
        w_width = bbox[2] - bbox[0]

        if w_width <= max_width:
            current = test
        else:
            lines.append(current)
            current = w

    lines.append(current)
    return lines


def make_meme_image(image: Image.Image, caption: str) -> Image.Image:
    """
    Render big meme-style text with strong stroke (outline).
    """
    if not caption:
        return image

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # BIGGER FONT SIZE (true meme style)
    base_font_size = max(int(h * 0.08), 24)

    # Try classic Impact font if available
    try:
        font = ImageFont.truetype("Impact.ttf", base_font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", base_font_size)
        except:
            font = ImageFont.load_default()

    # Wrap text to fit the width
    max_text_width = int(w * 0.95)
    lines = _wrap_text(draw, caption, font, max_text_width)

    # Line spacing
    line_height = base_font_size + 6
    total_text_height = line_height * len(lines)

    # Put at TOP with spacing
    y = max(int(h * 0.04), 10)

    # DRAW EACH LINE with thick black stroke + white fill
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (w - text_width) // 2

        # Black outline stroke for meme look
        draw.text(
            (x, y),
            line,
            font=font,
            fill="white",
            stroke_width=6,
            stroke_fill="black"
        )

        y += line_height

    return img


# --------------------------------------------------
# PREDICT FUNCTION FACTORY
# --------------------------------------------------
def make_predict_fn(adapter_dir: str, model_name: str):
    """
    Returns a function that Gradio will call.
    It takes a PIL image + extra tone text and calls our `generate` helper
    from infer.py, which expects:

        generate(
            image: Union[str, Image.Image],
            adapter_dir: Optional[str],
            model_name: str,
            num_return_sequences: int,
            max_new_tokens: int,
            extra_tone: str,
        )
    """

    def predict(image: Image.Image, extra_tone: str,
                num_captions: int, max_new_tokens: int):
        if image is None:
            return ["Please upload an image first."], None

        captions = generate_captions(
            image=image,
            adapter_dir=adapter_dir,
            model_name=model_name,
            num_return_sequences=int(num_captions),
            max_new_tokens=int(max_new_tokens),
            extra_tone=extra_tone,
        )

        # Build a meme image using the *first* caption (if any)
        if isinstance(captions, list) and len(captions) > 0:
            meme_img = make_meme_image(image, captions[0])
        else:
            meme_img = image

        return captions, meme_img

    return predict


# --------------------------------------------------
# GRADIO UI
# --------------------------------------------------
def build_interface(predict_fn):
    with gr.Blocks(title="Meme Generator – InstructBLIP + LoRA") as demo:
        gr.Markdown(
            "# 🧠🔥 LoRA Meme Generator\n"
            "Upload an image, optionally describe the **tone/vibes**, and get sarcastic captions.\n\n"
            "The first caption is also overlaid on the image to create a meme."
        )

        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Input image")
                tone_in = gr.Textbox(
                    label="Extra tone / vibes (optional)",
                    placeholder="e.g. tired student, broke but bougie, Monday morning energy…",
                )
                num_caps = gr.Slider(
                    minimum=1, maximum=5, step=1, value=3,
                    label="Number of captions"
                )
                max_tokens = gr.Slider(
                    minimum=8, maximum=64, step=4, value=32,
                    label="Max new tokens"
                )
                btn = gr.Button("Generate memes 💬")

            with gr.Column():
                out_text = gr.Textbox(
                    label="Generated captions",
                    lines=10,
                )
                out_img = gr.Image(
                    label="Meme preview (caption overlaid on image)",
                    type="pil"
                )

        def _wrapped_predict(image, extra_tone, num_captions, max_new_tokens):
            captions, meme_img = predict_fn(image, extra_tone, num_captions, max_new_tokens)

            if isinstance(captions, list):
                text_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(captions))
            else:
                text_block = str(captions)

            return text_block, meme_img

        btn.click(
            fn=_wrapped_predict,
            inputs=[img_in, tone_in, num_caps, max_tokens],
            outputs=[out_text, out_img],
        )

        return demo


# --------------------------------------------------
# CLI ENTRYPOINT
# --------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA run dir (e.g. outputs/meme_upgraded_round_2_run)",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/instructblip-flan-t5-xl",
        help="Base InstructBLIP model name",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio app",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    predict_fn = make_predict_fn(
        adapter_dir=args.adapter,
        model_name=args.model_name,
    )
    demo = build_interface(predict_fn)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
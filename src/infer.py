import argparse
import os
import re
import random
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, InstructBlipForConditionalGeneration
from peft import PeftModel


# --------------------------------------------------
# SARCASTIC TEMPLATES (safe but spicy)
# --------------------------------------------------

WHEN_TEMPLATES = [
    "realise adulthood is just side quests and no main plot.",
    "said 'it is what it is' but it is absolutely not what it is.",
    "thought things couldn't get worse and the universe said 'bet'.",
    "use 100% effort and it still looks like a tutorial level.",
    "remember you paid money to feel this stressed.",
    "are one inconvenience away from becoming the movie villain.",
    "call it 'character development' instead of a mental breakdown.",
    "thought you were the main character but the script got cancelled.",
]

ME_TEMPLATES = [
    "me pretending everything is fine while spiritually rage-quitting.",
    "me trying to act normal after one mildly embarrassing moment from 2014.",
    "me using 1% skill and 99% pure delusion.",
    "me saying 'this is the last time' for the fifteenth time this week.",
    "me processing the last three years in one single stare.",
]

POV_TEMPLATES = [
    "POV: you said 'how bad can it be?' and now you know.",
    "POV: you chose 'it'll be fun' instead of staying in bed.",
    "POV: your body said no but your ego said 'run it back'.",
    "POV: you’re realising the tutorial did NOT cover this situation.",
    "POV: you volunteered and immediately regretted volunteering.",
]

FALLBACK_TEMPLATES = [
    "this was not in the patch notes.",
    "life really said 'skill issue'.",
    "if coping was a full-time job.",
    "we laughing, but the damage is permanent.",
    "this is what 'I’m fine' actually looks like.",
]


# --------------------------------------------------
# STRIP DATASET EXPLAINER PHRASES
# --------------------------------------------------
def _strip_explainer_phrases(text: str) -> str:
    """
    Remove MEMECAP-style things like:
    - 'when you trying to convey that ...'
    - 'meme poster is ...'
    - 'the person who wrote the post is ...'
    """
    if not isinstance(text, str):
        text = str(text)

    t = text.strip()

    # generic 'trying to convey that'
    t = re.sub(
        r"\b(is|are|was|were)?\s*trying to convey that\b",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # "meme poster is", "the meme poster is"
    t = re.sub(
        r"\b(the\s+)?meme poster is\b",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # "the person who wrote the post is"
    t = re.sub(
        r"\bthe person who wrote the post is\b",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # Clean spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --------------------------------------------------
# POSTPROCESSING: hijack into meme templates
# --------------------------------------------------
def _postprocess_caption(text: str, max_words: int = 18) -> str:
    """
    Clean LLM output and then force it into meme style.
    If it starts with 'When you', 'When you're', 'Me:' or 'POV:',
    we override the tail with a handcrafted sarcastic template.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # strip outer quotes
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()

    text = _strip_explainer_phrases(text)
    low = text.lower()

    # Detect prefix
    prefix = None
    if low.startswith("when you're"):
        prefix = "When you're"
    elif low.startswith("when you"):
        prefix = "When you"
    elif low.startswith("me:"):
        prefix = "Me:"
    elif low.startswith("pov:"):
        prefix = "POV:"

    # If we got a meme-y prefix, snap to our templates
    if prefix is not None:
        if prefix.startswith("When you"):
            tail = random.choice(WHEN_TEMPLATES)
        elif prefix.startswith("When you're"):
            tail = random.choice(WHEN_TEMPLATES)
        elif prefix.startswith("Me:"):
            tail = random.choice(ME_TEMPLATES)
        elif prefix.startswith("POV:"):
            tail = random.choice(POV_TEMPLATES)
        caption = f"{prefix} {tail}"
    else:
        # Otherwise, keep the model text but still cap length,
        # then optionally replace with a fallback if too bland.
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
            text = " ".join(words)

        caption = text if text else random.choice(FALLBACK_TEMPLATES)

    # final length cap
    words = caption.split()
    if len(words) > max_words:
        caption = " ".join(words[:max_words])

    caption = caption.strip()

    # ensure punctuation
    if caption and caption[-1] not in ".!?":
        caption += "."

    # capitalise first letter if not prefixed by Me:/POV:
    if not caption.lower().startswith(("me:", "pov:")) and caption:
        caption = caption[0].upper() + caption[1:]

    return caption


# --------------------------------------------------
# VERY LIGHT DESCRIPTION FILTER
# --------------------------------------------------
def _looks_like_description(c: str) -> bool:
    """
    Only kill super-obvious alt-text / literal descriptions.
    """
    c_low = c.lower().strip()

    bad_starts = [
        "this image", "the image",
        "the picture", "in this picture",
        "image description", "this is a meme with the title",
        "caption:",
    ]
    if any(c_low.startswith(bs) for bs in bad_starts):
        return True

    # extreme alt-text pattern
    if c_low.startswith("a man ") or c_low.startswith("a woman ") or c_low.startswith("a person "):
        return True
    if c_low.startswith("the man ") or c_low.startswith("the woman ") or c_low.startswith("the person "):
        return True

    return False


# --------------------------------------------------
# GENERATION
# --------------------------------------------------
def generate(
    image: Union[str, Image.Image],
    adapter_dir: Optional[str] = None,
    model_name: str = "Salesforce/instructblip-flan-t5-xl",
    num_return_sequences: int = 3,
    max_new_tokens: int = 32,
    extra_tone: str = "",
) -> List[str]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------
    # Locate LoRA directory
    # ----------------------
    if adapter_dir:
        run_dir = adapter_dir
        lora_path = None

        for cand in ["lora_adapter_final", "lora_adapter", "adapter"]:
            cand_path = os.path.join(run_dir, cand)
            if os.path.isdir(cand_path):
                lora_path = cand_path
                break

        if lora_path is None:
            lora_path = run_dir   # fallback

        print(f"\n>>> Using LoRA run directory: {run_dir}")
        print(f">>> Detected LoRA adapter path: {lora_path}")

        processor_path = run_dir
    else:
        processor_path = model_name
        lora_path = None
        print("\n>>> No adapter provided. Running BASE model only.")

    # ----------------------
    # Load processor + base model
    # ----------------------
    processor = AutoProcessor.from_pretrained(processor_path)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # ----------------------
    # Load LoRA *into language_model*
    # ----------------------
    if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        print(">>> Loading LoRA into language_model...")
        try:
            model.language_model = PeftModel.from_pretrained(
                model.language_model,
                lora_path
            )
            print(">>> ✔ SUCCESS: LoRA loaded and attached!")
        except Exception as e:
            print(">>> ❌ ERROR loading LoRA:", e)
            print(">>> Running WITHOUT LoRA.")
    else:
        print(">>> ⚠ WARNING: No valid LoRA adapter found. Using BASE model.")

    model.to(device).eval()

    # ----------------------
    # Load image
    # ----------------------
    pil_image = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")

    # ----------------------
    # Meme prompt – push toward meme prefixes
    # ----------------------
    tone_snippet = f"\nExtra tone: {extra_tone.strip()}" if extra_tone else ""

    prompt = (
        "You are a chaotic, sarcastic internet meme creator.\n"
        "Write ONE short meme caption for THIS image.\n"
        "RULES:\n"
        "- The caption is the text you would put on the meme.\n"
        "- Do NOT literally describe what is happening in the image.\n"
        "- React to the emotion, situation, or vibe instead.\n"
        "- Prefer starting with 'When you', 'When you're', 'Me:', or 'POV:'.\n"
        "- Be funny, dark, ironic, a bit unhinged (no slurs, no self-harm).\n"
        "- Speak like a Reddit/Twitter shitposter.\n"
        "- Max 18 words. No emojis, no hashtags, no quotation marks.\n"
        f"{tone_snippet}\n"
        "Caption:"
    )

    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

    # ----------------------
    # Generate
    # ----------------------
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.9,
            temperature=1.05,
            top_k=50,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.15,
            num_return_sequences=num_return_sequences,
        )

    raw = processor.batch_decode(outputs, skip_special_tokens=True)

    # ----------------------
    # Filter + clean
    # ----------------------
    captions: List[str] = []
    seen = set()

    for c in raw:
        if not c:
            continue
        if _looks_like_description(c):
            continue
        c = _postprocess_caption(c)
        if c and c not in seen:
            seen.add(c)
            captions.append(c)

    # fallback if *everything* got filtered out
    if len(captions) == 0:
        for c in raw:
            c = _postprocess_caption(c)
            if c and c not in seen:
                captions.append(c)

    return captions


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--num_return_sequences", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--tone", type=str, default="")
    args = ap.parse_args()

    caps = generate(
        image=args.image,
        adapter_dir=args.adapter,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
        extra_tone=args.tone,
    )

    for i, c in enumerate(caps, 1):
        print(f"[i] {c}")


if __name__ == "__main__":
    main()
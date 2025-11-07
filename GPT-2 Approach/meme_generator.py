import os
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import streamlit as st
from textwrap import wrap

# Set up Streamlit interface
st.title("Adaptive Meme Generator")
st.write("Upload an image and let the AI generate a dark humor meme for it!")

# Step 1: Upload Image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Step 2: Generate Image Description
    st.write("Analyzing the image...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=img, return_tensors="pt")
    outputs = model.generate(**inputs)
    image_description = processor.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Image Description: {image_description}")

    # Step 3: Generate Meme Caption
    st.write("Generating meme text...")

    meme_generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

    def generate_meme_text(description):
        prompt = (
            f"Write a short, funny, and sarcastic dark humor meme for an image showing {description}. "
            "Keep it under 15 words, make it hilarious, and avoid links or irrelevant text:"
            "Do not use external link, just add a straight to the point funny and sarcastic caption."
        )
        result = meme_generator(
            prompt,
            max_new_tokens=25,  # Allow the model to generate up to 25 new tokens
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.9,  # Encourage diverse and creative outputs
            temperature=1.0,  # Balance randomness
        )
        # Extract and clean up the generated text
        generated_text = result[0]["generated_text"]
        meme_text = generated_text.replace(prompt, "").strip()
        
        # Ensure the caption doesn't exceed 2 lines
        if len(meme_text.split()) > 15:  # Arbitrary word count limit
            meme_text = " ".join(meme_text.split()[:15]) + "..."
        
        return meme_text


    meme_caption = generate_meme_text(image_description)
    st.write(f"Meme Caption: {meme_caption}")

    # Step 4: Overlay Meme Text on Image
    def add_text_to_image(image, text, output_path="meme_output.jpg"):
        if image.mode != "RGB":
            image = image.convert("RGB")

        draw = ImageDraw.Draw(image)
        font_path = "arial.ttf"  # Provide a valid .ttf font path
        try:
            font = ImageFont.truetype(font_path, 40)
        except IOError:
            st.write("Font not found, using default font.")
            font = ImageFont.load_default()

        max_width = image.width - 40  # Keep a margin
        max_height = image.height * 0.7  # Occupy 70% of the image height

        while True:
            wrapped_text = wrap(text, width=int(max_width / font.getlength("A")))
            total_height = sum(
                [
                    draw.textbbox((0, 0), line, font=font)[3]
                    - draw.textbbox((0, 0), line, font=font)[1]
                    for line in wrapped_text
                ]
            )
            if total_height <= max_height and font.size > 10:
                break
            font = ImageFont.truetype(font_path, font.size - 2)

        # Center text vertically and horizontally
        y = (image.height - total_height) // 2
        for line in wrapped_text:
            text_width = draw.textlength(line, font=font)
            x = (image.width - text_width) // 2
            draw.text((x, y), line, fill="white", font=font, stroke_width=2, stroke_fill="black")
            y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]

        image.save(output_path)
        return output_path

    st.write("Creating your meme...")
    output_path = "meme_output.jpg"
    add_text_to_image(img.copy(), meme_caption, output_path)

    # Display the final meme
    st.image(output_path, caption="Generated Meme", use_container_width=True)
    st.write("Download your meme below!")
    with open(output_path, "rb") as file:
        st.download_button(label="Download Meme", data=file, file_name="meme_output.jpg", mime="image/jpeg")

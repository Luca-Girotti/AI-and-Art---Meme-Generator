from PIL import Image
from transformers import BlipProcessor, TFBlipForConditionalGeneration

# Path to the fine-tuned model
model_path = "./blip-finetuned-epoch-5"

# Load the fine-tuned model and processor
model = TFBlipForConditionalGeneration.from_pretrained(model_path)
processor = BlipProcessor.from_pretrained(model_path)

# Load an image for prediction
image_path = "33e92f.jpg"
image = Image.open(image_path).convert("RGB")

# Prepare the inputs
inputs = processor(images=image, return_tensors="tf")

# Generate a caption with randomness
outputs = model.generate(
    pixel_values=inputs["pixel_values"],
    min_length=2,               # Set minimum length of the caption
    max_length=50,               # Set maximum length of the caption
    top_k=10,                    # Consider the top 10 probable tokens
    do_sample=True               # Enable sampling to add randomness
)

# Decode the generated caption
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Generated Caption: {caption}")

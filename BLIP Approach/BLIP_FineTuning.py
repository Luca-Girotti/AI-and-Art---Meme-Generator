import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
#print to check if the gpu is available
print(torch.cuda.is_available())


# Load the image captioning dataset
labels_csv_path = "labels.csv"
df = pd.read_csv(labels_csv_path)
text_corrected = df['text_corrected']
print(f"Loaded {len(text_corrected)} captions.")

# Load and sort images
images = []
for file_name in os.listdir("images"):
    try:
        img_path = os.path.join("images", file_name)
        with Image.open(img_path) as img:
            images.append((img, file_name))
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

print(f"Loaded {len(images)} images.")
ordered_images = sorted(images, key=lambda x: int(x[1].split('_')[1].split('.')[0]))
ordered_images = [file_name for _, file_name in ordered_images]
print(f"Ordered {len(ordered_images)} images.")

# Filter and ensure captions are strings
valid_images = []
valid_captions = []

for file_name, caption in zip(ordered_images, text_corrected):
    # Check if caption is valid (not null, empty, or non-string)
    if isinstance(caption, str) and caption.strip():
        valid_images.append(file_name)
        valid_captions.append(caption.strip())
    
ordered_images = valid_images
text_corrected = valid_captions

print(f"Filtered captions: Retained {len(valid_captions)} valid captions.")
print(f"Filtered images: Retained {len(ordered_images)} valid images.")

# Initialize the processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Define a custom PyTorch Dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, image_filenames, captions, processor):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        caption = self.captions[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=100,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0),
        }

# Prepare dataset and dataloaders
batch_size = 4
image_dir = "images"

dataset = ImageCaptionDataset(image_dir, ordered_images, text_corrected, processor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Load the model and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.train()

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training and validation steps
def train_step(batch):
    optimizer.zero_grad()
    outputs = model(
        pixel_values=batch["pixel_values"].to(device),
        input_ids=batch["input_ids"].to(device),
        labels=batch["labels"].to(device),
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.item()

def val_step(batch):
    with torch.no_grad():
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
    return outputs.loss.item()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training
    model.train()
    train_loss = 0.0
    total_batches = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        train_loss += train_step(batch)

        # Display progress percentage
        progress = (batch_idx + 1) / total_batches * 100
        print(f"Training Progress: {progress:.2f}%", end="\r")

    # Validation
    model.eval()
    val_loss = 0.0
    total_val_batches = len(val_loader)
    for batch_idx, batch in enumerate(val_loader):
        val_loss += val_step(batch)
        


    # Save the model after each epoch
    epoch_save_path = f"./blip-finetuned-epoch-{epoch + 1}"
    os.makedirs(epoch_save_path, exist_ok=True)
    model.save_pretrained(epoch_save_path)
    processor.save_pretrained(epoch_save_path)
    print(f"Model saved after epoch {epoch + 1} at {epoch_save_path}")


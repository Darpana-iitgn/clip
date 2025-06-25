import os
import torch
import clip
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== 1. Load CLIP Model ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ==== 2. Load MNIST Data (only digit 9) ====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.Resize(224),                      # Resize for CLIP
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_root = "/home/project/IntCLIP/clip/mnist_data"
mnist = datasets.MNIST(root=dataset_root, train=False, download=False, transform=transform)

# Filter images with label 9
images_of_nine = [img for img, label in mnist if label == 9]
print(f"Found {len(images_of_nine)} images of digit 9.")

# Create a batch tensor for CLIP input
image_input = torch.stack(images_of_nine).to(device)

# ==== 3. Encode Images in Batches ====
batch_size = 128
image_features = []

with torch.no_grad():
    for i in tqdm(range(0, len(image_input), batch_size), desc="Encoding images"):
        batch = image_input[i:i+batch_size]
        feats = model.encode_image(batch)
        feats /= feats.norm(dim=-1, keepdim=True)
        image_features.append(feats)

image_features = torch.cat(image_features, dim=0)

# ==== 4. Define Prompts ====
prompts = [
    "a 45-degree rotated image",
    "a 135-degree rotated image",
    "a image which doesn't look like nine",
    "a right-shifted image",
    "a zoomed-out image",
    "a slightly curved image",
    "a blurry image"
]

# Encode prompts
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ==== 5. Find Top Images per Prompt ====
similarity = text_features @ image_features.T  # [num_prompts x num_images]
top_n = 5
output_dir = "clip_prompt_results"
os.makedirs(output_dir, exist_ok=True)

for i, prompt in enumerate(prompts):
    top_indices = similarity[i].topk(top_n).indices
    top_images = [images_of_nine[j].permute(1, 2, 0).cpu() * 0.5 + 0.5 for j in top_indices]

    # Show and save results
    fig, axes = plt.subplots(1, top_n, figsize=(15, 3))
    for ax, img in zip(axes, top_images):
        ax.imshow(img.numpy())
        ax.axis("off")
    plt.suptitle(f"Prompt: '{prompt}'", fontsize=14)
    plt.tight_layout()

    # Save figure
    filename = prompt.replace(" ", "_").replace("-", "") + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
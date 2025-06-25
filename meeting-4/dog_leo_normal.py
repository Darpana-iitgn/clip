import os
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.manifold import TSNE

# ---------------------------
# Global Parameters and Paths
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
TEMPERATURE = 0.07
JOINT_EMBED_DIM = 256
DATA_DIR = os.path.expanduser("~/dataset/ImageNet10")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class directories mapping:
# "757": leopard, "936": dog
CLASS_DIRS = {
    "757": os.path.join(DATA_DIR, "n02128757"),
    "936": os.path.join(DATA_DIR, "n02085936")
}

# ---------------------------
# Define Transforms
# ---------------------------
# Training transform: resize and convert to tensor
train_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor()
])

# Evaluation transform: resize and tensor conversion
eval_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor()
])

# Custom augmentations for testing (all expect PIL Images)
left_shift_45_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=45, translate=(-3, 0), scale=1, shear=0)),
    transforms.ToTensor()
])
right_shift_45_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=45, translate=(3, 0), scale=1, shear=0)),
    transforms.ToTensor()
])
left_shift_135_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=135, translate=(-3, 0), scale=1, shear=0)),
    transforms.ToTensor()
])
right_shift_135_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=135, translate=(3, 0), scale=1, shear=0)),
    transforms.ToTensor()
])

# ---------------------------
# Dataset Definitions
# ---------------------------
class DogLeopardDataset(Dataset):
    """
    Dataset for an ImageNet10 structure with two classes:
      - "757": leopard
      - "936": dog
    """
    def __init__(self, class_dirs, transform=None):
        self.transform = transform
        self.samples = []
        self.class_mapping = {"757": "leopard", "936": "dog"}
        for class_id, class_dir in class_dirs.items():
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Directory {class_dir} not found.")
            class_images = [
                os.path.join(class_dir, fname)
                for fname in os.listdir(class_dir)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.samples.extend([(img_path, self.class_mapping[class_id])
                                 for img_path in class_images])
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Custom dataset for test-time augmentations (4 types)
class CustomTestDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Ensure image is a PIL Image for augmentation transforms.
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.cpu())
        # Divide the dataset into 4 equal parts based on index.
        n = len(self.dataset)
        if idx < n // 4:
            aug_img = left_shift_45_transform(image)
            aug_label = label + "_left45"
        elif idx < 2 * n // 4:
            aug_img = right_shift_45_transform(image)
            aug_label = label + "_right45"
        elif idx < 3 * n // 4:
            aug_img = left_shift_135_transform(image)
            aug_label = label + "_left135"
        else:
            aug_img = right_shift_135_transform(image)
            aug_label = label + "_right135"
        return aug_img, aug_label, label

# A simple wrapper to apply evaluation transforms to a subset.
class TransformWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.cpu())
        return self.transform(image), label

# ---------------------------
# Text Processing
# ---------------------------
all_texts = ["dog", "leopard"]
vocab = sorted(set("".join(all_texts)))
stoi = {ch: i+1 for i, ch in enumerate(vocab)}  # reserve 0 for padding
vocab_size = len(stoi) + 1

def encode_text(s, max_len=15):
    tokens = [stoi.get(ch, 0) for ch in s]
    if len(tokens) < max_len:
        tokens += [0]*(max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long)

# ---------------------------
# Model Components
# ---------------------------
print("Loading DINOv2 ViT-B/14 model...")
dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
dino_model.to(device)
dino_model.eval()

with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(device)
    features = dino_model(dummy)
    img_feat_dim = features.shape[-1]
print(f"DINOv2 feature dimension: {img_feat_dim}")

class ImageProjection(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, proj_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, proj_dim)
    def forward(self, token_sequences):
        emb = self.embedding(token_sequences).mean(dim=1)
        return F.normalize(self.proj(emb), dim=-1)

class MultiModalModel(nn.Module):
    def __init__(self, dino_model, img_feat_dim, joint_dim, vocab_size, text_embed_dim=64):
        super().__init__()
        self.image_encoder = dino_model
        self.image_proj = ImageProjection(img_feat_dim, joint_dim)
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim, joint_dim)

    def forward(self, images, text_tokens):
        with torch.no_grad():
            img_features = self.image_encoder(images)
        return self.image_proj(img_features), self.text_encoder(text_tokens)

model = MultiModalModel(dino_model, img_feat_dim, JOINT_EMBED_DIM, vocab_size)
model.to(device)

def contrastive_loss(img_emb, txt_emb, temperature):
    logits = (img_emb @ txt_emb.T) * temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------------------
# Data Loading and Stratified Splitting
# ---------------------------
full_dataset = DogLeopardDataset(CLASS_DIRS, transform=train_transform)
total_images = len(full_dataset)
print("Total images:", total_images)

# Create separate lists of indices for each class.
indices_leopard = [i for i, (_, label) in enumerate(full_dataset) if label == "leopard"]
indices_dog = [i for i, (_, label) in enumerate(full_dataset) if label == "dog"]

# We require at least 1300 images per class:
if len(indices_leopard) < 1300 or len(indices_dog) < 1300:
    raise ValueError("Not enough images per class for stratified splitting!")

random.shuffle(indices_leopard)
random.shuffle(indices_dog)

# For each class: train = 1000, validation = 200, test = 100
train_indices_leopard = indices_leopard[:1000]
val_indices_leopard   = indices_leopard[1000:1200]
test_indices_leopard  = indices_leopard[1200:1300]

train_indices_dog = indices_dog[:1000]
val_indices_dog   = indices_dog[1000:1200]
test_indices_dog  = indices_dog[1200:1300]

train_indices = train_indices_leopard + train_indices_dog
val_indices   = val_indices_leopard  + val_indices_dog
test_indices  = test_indices_leopard + test_indices_dog

train_dataset = Subset(full_dataset, train_indices)
val_dataset   = Subset(full_dataset, val_indices)
test_dataset  = Subset(full_dataset, test_indices)

# Apply evaluation transform to validation and test sets.
val_dataset  = TransformWrapper(val_dataset, eval_transform)
test_dataset = TransformWrapper(test_dataset, eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def print_class_distribution(dataset, name):
    counts = {}
    for _, label in dataset:
        counts[label] = counts.get(label, 0) + 1
    print(f"{name} distribution:", counts)

print_class_distribution(train_dataset, "Train")
print_class_distribution(val_dataset, "Validation")
print_class_distribution(test_dataset, "Test")

# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} training"):
        images = images.to(device)
        text_tokens = torch.stack([encode_text(label) for label in labels]).to(device)
        optimizer.zero_grad()
        img_emb, txt_emb = model(images, text_tokens)
        loss = contrastive_loss(img_emb, txt_emb, TEMPERATURE)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} validation"):
            images = images.to(device)
            text_tokens = torch.stack([encode_text(label) for label in labels]).to(device)
            img_emb, txt_emb = model(images, text_tokens)
            val_loss += contrastive_loss(img_emb, txt_emb, TEMPERATURE).item() * images.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

# ---------------------------
# Evaluation and t-SNE Visualization for Images
# ---------------------------
custom_test_dataset = CustomTestDataset(test_dataset)
custom_test_loader = DataLoader(custom_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

all_image_embeddings = []
all_aug_labels = []   
all_base_labels = []  
model.eval()
with torch.no_grad():
    for images, aug_labels, base_labels in tqdm(custom_test_loader, desc="Extracting image embeddings"):
        images = images.to(device)
        text_tokens = torch.stack([encode_text(label) for label in base_labels]).to(device)
        img_emb, _ = model(images, text_tokens)
        all_image_embeddings.append(img_emb.cpu().numpy())
        all_aug_labels.extend(aug_labels)
        all_base_labels.extend(base_labels)
all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)

def plot_tsne_images(image_embeddings, aug_labels):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(image_embeddings)
    
    plt.figure(figsize=(14, 10))
    leopard_colors = {
        "left45": "#8da0cb",
        "right45": "#1f78b4",
        "left135": "#b2abd2",
        "right135": "#6a3d9a"
    }
    dog_colors = {
        "left45": "#ffbb78",
        "right45": "#ff7f0e",
        "left135": "#d62728",
        "right135": "#8c564b"
    }
    
    for i, label in enumerate(aug_labels):
        base, aug = label.split('_')
        color = leopard_colors[aug] if base == "leopard" else dog_colors[aug]
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                    color=color, marker='o', s=15, alpha=0.7)
    
    plt.title("t-SNE of Image Embeddings (Augmentation Types)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    legend_elements = []
    for aug, color in leopard_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                               label=f"Leopard {aug}", markerfacecolor=color, markersize=10))
    for aug, color in dog_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                               label=f"Dog {aug}", markerfacecolor=color, markersize=10))
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "normal_dog_leopard_images.png"), bbox_inches='tight')
    plt.show()


plot_tsne_images(all_image_embeddings, all_aug_labels)

# ---------------------------
# t-SNE Visualization for Text
# ---------------------------
# Create text prompts for each augmentation variant (4 types) for both classes.
text_prompts = []
for base in ["leopard", "dog"]:
    for aug in ["left45", "right45", "left135", "right135"]:
        text_prompts.append(f"{base}_{aug}")

def generate_text_embeddings(model, labels):
    model.eval()
    text_embeddings = []
    with torch.no_grad():
        for label in labels:
            token_seq = encode_text(label).unsqueeze(0).to(device)
            emb = model.text_encoder(token_seq)
            text_embeddings.append(emb.cpu().numpy().flatten())
    return np.array(text_embeddings)

text_embeddings = generate_text_embeddings(model, text_prompts)

def plot_tsne_text(text_embeddings, text_labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    embeddings_2d = tsne.fit_transform(text_embeddings)
    
    plt.figure(figsize=(14, 10))
    leopard_colors = {
        "left45": "#8da0cb",
        "right45": "#1f78b4",
        "left135": "#b2abd2",
        "right135": "#6a3d9a"
    }
    dog_colors = {
        "left45": "#ffbb78",
        "right45": "#ff7f0e",
        "left135": "#d62728",
        "right135": "#8c564b"
    }
    
    for i, label in enumerate(text_labels):
        base, aug = label.split('_')
        color = leopard_colors[aug] if base == "leopard" else dog_colors[aug]
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                    color=color, marker='*', s=100)
    
    plt.title("t-SNE of Text Embeddings (Augmentation Types)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    legend_elements = []
    for aug, color in leopard_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                               label=f"Leopard {aug}", markerfacecolor=color, markersize=15))
    for aug, color in dog_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                               label=f"Dog {aug}", markerfacecolor=color, markersize=15))
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "normal_dog_leopard_text.png"), bbox_inches='tight')
    plt.show()


plot_tsne_text(text_embeddings, text_prompts)

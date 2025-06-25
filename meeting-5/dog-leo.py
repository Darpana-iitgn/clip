import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms, datasets
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
TEMPERATURE = 0.07
JOINT_EMBED_DIM = 256

# Dataset paths
DATA_DIR = os.path.expanduser("~/dataset/ImageNet10")
OUTPUT_DIR = "outputs-1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class directories mapping:
# "757": leopard, "936": dog
CLASS_DIRS = {
    "757": os.path.join(DATA_DIR, "n02128757"),
    "936": os.path.join(DATA_DIR, "n02085936")
}

# --- DogLeopardDataset definition (unchanged except numeric labels) ---
class DogLeopardDataset(Dataset):
    """
    Dataset for an ImageNet10 structure with two classes:
      - "757": leopard
      - "936": dog
    Returns numeric labels: 1 for leopard, 0 for dog.
    """
    def __init__(self, class_dirs, transform=None):
        self.transform = transform
        self.samples = []
        # Numeric mapping to match the rest of the code:
        self.class_mapping = {"757": 1, "936": 0}
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
# --- End of DogLeopardDataset ---

# Define transforms: all output images are resized to 224x224
normal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

left_shift_45_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=45, translate=(-3, 0), scale=1, shear=0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

right_shift_45_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=45, translate=(3, 0), scale=1, shear=0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

left_shift_135_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=135, translate=(-3, 0), scale=1, shear=0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

right_shift_135_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=135, translate=(3, 0), scale=1, shear=0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom Train Dataset (ensures images are PIL Images)
class CustomTrainDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Convert image to PIL Image if needed.
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        n = len(self.dataset)
        # Divide dataset into 5 parts to apply different transforms
        if idx < n // 5:
            return normal_transform(image), label
        elif idx < 2 * n // 5:
            return left_shift_45_transform(image), label
        elif idx < 3 * n // 5:
            return right_shift_45_transform(image), label
        elif idx < 4 * n // 5:
            return left_shift_135_transform(image), label
        else:
            return right_shift_135_transform(image), label

# Custom Test Dataset (returns augmented image, augmentation label, and base label)
class CustomTestDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        # Use modulo to choose augmentation for each sample regardless of absolute index.
        mod = idx % 5
        # Convert numeric label to text using the global label_names mapping.
        label_text = label_names[label]  # label_names is defined later as {0: "dog", 1: "leopard"}
        if mod == 0:
            aug_img = left_shift_45_transform(image)
            aug_label = f"{label_text}_left45"
        elif mod == 1:
            aug_img = right_shift_45_transform(image)
            aug_label = f"{label_text}_right45"
        elif mod == 2:
            aug_img = left_shift_135_transform(image)
            aug_label = f"{label_text}_left135"
        elif mod == 3:
            aug_img = right_shift_135_transform(image)
            aug_label = f"{label_text}_right135"
        else:
            aug_img = normal_transform(image)
            aug_label = f"{label_text}_normal"
        return aug_img, aug_label, label

# Use DogLeopardDataset to load data from the correct folders.
full_dataset = DogLeopardDataset(CLASS_DIRS, transform=None)

# Separate indices for each class (assuming class 0: dog, class 1: leopard)
dog_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
leopard_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]

np.random.seed(42)
np.random.shuffle(dog_indices)
np.random.shuffle(leopard_indices)

train_dog = dog_indices[:1000]
val_dog   = dog_indices[1000:1200]
test_dog  = dog_indices[1200:1300]

train_leopard = leopard_indices[:1000]
val_leopard   = leopard_indices[1000:1200]
test_leopard  = leopard_indices[1200:1300]

train_indices = train_dog + train_leopard
val_indices   = val_dog + val_leopard
test_indices  = test_dog + test_leopard

train_dataset = Subset(full_dataset, train_indices)
val_dataset   = Subset(full_dataset, val_indices)
test_dataset  = Subset(full_dataset, test_indices)

custom_train_dataset = CustomTrainDataset(train_dataset)
custom_val_dataset   = CustomTestDataset(val_dataset)
custom_test_dataset  = CustomTestDataset(test_dataset)

train_loader = DataLoader(custom_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(custom_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(custom_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Image Encoder using Adaptive Pooling
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (B,64,224,224)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B,64,112,112)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (B,128,112,112)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B,128,56,56)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# (B,256,56,56)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (B,256,28,28)
            nn.AdaptiveAvgPool2d((7, 7))                           # (B,256,7,7)
        )
        self.fc = nn.Linear(256 * 7 * 7, JOINT_EMBED_DIM)

    def forward(self, x):
        x = self.conv(x)
        # Uncomment to debug shape: print("Image encoder conv output shape:", x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, dim=-1)

# Text Encoder (for our single-token texts)
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, JOINT_EMBED_DIM)
        )

    def forward(self, x):
        # x shape: (batch_size, 1)
        x = self.embedding(x)  # shape: (batch_size, 1, embed_dim)
        x = torch.mean(x, dim=1)  # shape: (batch_size, embed_dim)
        x = self.fc(x)
        return nn.functional.normalize(x, dim=-1)

# CLIP Model
class CLIP(nn.Module):
    def __init__(self, vocab_size):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(vocab_size)

    def forward(self, image, text):
        image_emb = self.image_encoder(image)
        text_emb = self.text_encoder(text)
        return image_emb, text_emb

# Vocabulary and helper to encode text
vocab = {"dog": 0, "leopard": 1}
vocab_size = len(vocab)

def encode_text(text_list):
    """
    Given a list of strings (e.g., ["dog", "leopard"]), returns a tensor of shape (batch_size, 1).
    """
    indices = [[vocab[t]] for t in text_list]
    return torch.tensor(indices, dtype=torch.long)

# Loss Function: Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_emb, text_emb):
        logits = torch.matmul(image_emb, text_emb.T) / self.temperature
        labels = torch.arange(image_emb.size(0)).to(image_emb.device)
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

model = CLIP(vocab_size).to(device)
criterion = ContrastiveLoss(TEMPERATURE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Mapping for label names (numeric label -> text label)
label_names = {0: "dog", 1: "leopard"}

def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        # Convert each tensor label to a Python number using .item()
        text = encode_text([label_names[label.item()] for label in labels]).to(device)
        optimizer.zero_grad()
        image_emb, text_emb = model(images, text)
        loss = criterion(image_emb, text_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss / len(train_loader)

def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, aug_labels, labels in val_loader:
            images = images.to(device)
            text = encode_text([label_names[label.item()] for label in labels]).to(device)
            image_emb, text_emb = model(images, text)
            loss = criterion(image_emb, text_emb)
            total_loss += loss.item()
    return total_loss / len(val_loader)

for epoch in range(1, EPOCHS + 1):
    train_loss = train(epoch)
    val_loss = validate()
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# Evaluation and t-SNE Visualization for Images
custom_test_loader = DataLoader(custom_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
all_image_embeddings = []
all_aug_labels = []   # e.g., "leopard_left45", "dog_normal", etc.
all_base_labels = []  # e.g., 0 or 1
model.eval()
with torch.no_grad():
    for images, aug_labels, base_labels in tqdm(custom_test_loader, desc="Extracting image embeddings"):
        images = images.to(device)
        text_tokens = torch.stack([encode_text([label_names[label.item()]]) for label in base_labels]).to(device)
        img_emb, _ = model(images, text_tokens)
        all_image_embeddings.append(img_emb.cpu().numpy())
        all_aug_labels.extend(aug_labels)
        all_base_labels.extend(base_labels)
all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)

def plot_tsne_images(image_embeddings, aug_labels):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(image_embeddings)
    
    plt.figure(figsize=(14, 10))
    # Add a "normal" key for both classes.
    leopard_colors = {"left45": "#8da0cb", "right45": "#1f78b4",
                      "left135": "#b2abd2", "right135": "#6a3d9a",
                      "normal": "#d73027"}
    dog_colors = {"left45": "#ffbb78", "right45": "#ff7f0e",
                  "left135": "#d62728", "right135": "#8c564b",
                  "normal": "#1a9850"}
    
    for i, label in enumerate(aug_labels):
        base, aug = label.split('_')
        if base == "leopard":
            color = leopard_colors.get(aug, "#000000")
        else:
            color = dog_colors.get(aug, "#000000")
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
    plt.savefig(os.path.join(OUTPUT_DIR, "dog_leopard_images.png"), bbox_inches='tight')
    plt.show()

plot_tsne_images(all_image_embeddings, all_aug_labels)

# t-SNE Visualization for Text
text_prompts = []
for base in ["leopard", "dog"]:
    for aug in ["left45", "right45", "left135", "right135"]:
        text_prompts.append(f"{base}_{aug}")

def generate_text_embeddings(model, labels):
    model.eval()
    text_embeddings = []
    with torch.no_grad():
        for label in labels:
            base_label = label.split('_')[0]
            token_seq = encode_text([base_label]).unsqueeze(0).to(device)
            emb = model.text_encoder(token_seq)
            text_embeddings.append(emb.cpu().numpy().flatten())
    return np.array(text_embeddings)

text_embeddings = generate_text_embeddings(model, text_prompts)

def plot_tsne_text(text_embeddings, text_labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    embeddings_2d = tsne.fit_transform(text_embeddings)
    
    plt.figure(figsize=(14, 10))
    leopard_colors = {"left45": "#8da0cb", "right45": "#1f78b4",
                      "left135": "#b2abd2", "right135": "#6a3d9a"}
    dog_colors = {"left45": "#ffbb78", "right45": "#ff7f0e",
                  "left135": "#d62728", "right135": "#8c564b"}
    
    for i, label in enumerate(text_labels):
        base, aug = label.split('_')
        if base == "leopard":
            color = leopard_colors.get(aug, "#000000")
        else:
            color = dog_colors.get(aug, "#000000")
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
    plt.savefig(os.path.join(OUTPUT_DIR, "dog_leopard_text.png"), bbox_inches='tight')
    plt.show()

plot_tsne_text(text_embeddings, text_prompts)

leopard_indices_count = [i for i, label in enumerate(all_aug_labels) if "leopard" in label]
print(f"Number of leopard samples: {len(leopard_indices_count)}")

from collections import Counter
aug_label_counter = Counter(all_aug_labels)
print("Augmented Label Distribution:", aug_label_counter)
import torch
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Filter for 0s and 1s
x_train_0 = x_train[(y_train == 0)]
y_train_0 = y_train[(y_train == 0)]
x_train_1 = x_train[(y_train == 1)]
y_train_1 = y_train[(y_train == 1)]

x_test_0 = x_test[(y_test == 0)]
y_test_0 = y_test[(y_test == 0)]
x_test_1 = x_test[(y_test == 1)]
y_test_1 = y_test[(y_test == 1)]

x_train = np.concatenate([x_train_0, x_train_1], axis=0)
y_train = np.concatenate([y_train_0, y_train_1], axis=0)
x_test = np.concatenate([x_test_0, x_test_1], axis=0)
y_test = np.concatenate([y_test_0, y_test_1], axis=0)


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

num_to_word = {
    0: 'zero',
    1: 'one'
}

text_labels_train = [f"{num_to_word[i]}" for i in y_train]
text_labels_test = [f"{num_to_word[i]}" for i in y_test]

class MNISTDataset(Dataset):
    def __init__(self, images, labels, text_labels, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.text_labels = text_labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        text_label = self.text_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, text_label

# Image transforms
normal_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze().cpu().numpy()),
    transforms.ToPILImage(),
    transforms.ToTensor()
])

left_shift_45_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze().cpu().numpy()),
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=45, translate=(-3, 0), scale=1, shear=0)),
    transforms.ToTensor()
])

right_shift_45_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze().cpu().numpy()),
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=45, translate=(3, 0), scale=1, shear=0)),
    transforms.ToTensor()
])

left_shift_135_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze().cpu().numpy()),
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=135, translate=(-3, 0), scale=1, shear=0)),
    transforms.ToTensor()
])

right_shift_135_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze().cpu().numpy()),
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=135, translate=(3, 0), scale=1, shear=0)),
    transforms.ToTensor()
])

class CustomTestDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, text_label = self.dataset[idx]
        if idx < len(self.dataset) // 5:
            return left_shift_45_transform(image), label, text_label
        elif idx < 2 * len(self.dataset) // 5:
            return right_shift_45_transform(image), label, text_label
        elif idx < 3 * len(self.dataset) // 5:
            return left_shift_135_transform(image), label, text_label
        elif idx < 4 * len(self.dataset) // 5:
            return right_shift_135_transform(image), label, text_label
        else:
            return normal_transform(image), label, text_label

# Data loading
batch_size = 64
train_dataset = MNISTDataset(x_train, y_train, text_labels_train)
test_dataset = MNISTDataset(x_test, y_test, text_labels_test)
custom_train_dataset = CustomTestDataset(train_dataset)
custom_test_dataset = CustomTestDataset(test_dataset)
train_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=False)

# Text processing
chars = sorted(list(set("".join(text_labels_train))))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Model architecture
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64*7*7, 10)
        
    def forward(self, x):
        x = self.conv(x).flatten(1)
        return nn.functional.normalize(self.fc(x), dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        
    def forward(self, x):
        return nn.functional.normalize(self.fc(self.embedding(x).mean(1)), dim=-1)

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(vocab_size)
        
    def forward(self, image, text):
        return self.image_encoder(image), self.text_encoder(text)

# Training setup
clip_model = CLIP().to(device)
optimizer = optim.Adam(clip_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    clip_model.train()
    total_loss = total_acc = 0
    
    for images, _, text_labels in train_loader:
        images = images.to(device)
        text_seq = nn.utils.rnn.pad_sequence(
            [torch.tensor(encode(label)) for label in text_labels],
            batch_first=True
        ).to(device)
        
        optimizer.zero_grad()
        img_emb, txt_emb = clip_model(images, text_seq)
        logits = (img_emb @ txt_emb.T) * 5  # Temperature 0.2
        loss = criterion(logits, torch.arange(len(images)).to(device))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += (logits.argmax(1) == torch.arange(len(images)).to(device)).float().mean().item()
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {total_acc/len(train_loader):.4f}")

# t-SNE Plotting
def plot_tsne(image_embeddings, text_embeddings, image_labels, text_labels):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(np.vstack([image_embeddings, text_embeddings]))
    
    plt.figure(figsize=(12,8))
    # Image embeddings
    colors = {
    '0_normal': '#1f77b4', '0_left45': '#aec7e8', '0_right45': '#17becf', 
    '0_left135': '#7f7f7f', '0_right135': '#c7c7c7',
    '1_normal': '#d62728', '1_left45': '#ff9896', '1_right45': '#ff7f0e',
    '1_left135': '#8c564b', '1_right135': '#e377c2'
}

    for i, label in enumerate(image_labels):
        plt.scatter(*embeddings[i], c=colors[label], marker='o' if '0' in label else '^', s=50, alpha=0.7)
    
    # Text embeddings
    text_colors = {'zero': '#2ca02c', 'one': '#9467bd'}
    for i, label in enumerate(text_labels, len(image_embeddings)):
        plt.scatter(*embeddings[i], c=text_colors[label], marker='*', s=200)
    
    legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='0 Normal'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#aec7e8', markersize=10, label='0 Left 45°'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#17becf', markersize=10, label='0 Right 45°'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f', markersize=10, label='0 Left 135°'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#c7c7c7', markersize=10, label='0 Right 135°'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#d62728', markersize=10, label='1 Normal'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff9896', markersize=10, label='1 Left 45°'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', markersize=10, label='1 Right 45°'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#8c564b', markersize=10, label='1 Left 135°'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#e377c2', markersize=10, label='1 Right 135°'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ca02c', markersize=15, label='"zero"'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#9467bd', markersize=15, label='"one"')
]

    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("t-SNE of CLIP Embeddings (0s and 1s)")
    plt.xlabel("Dimension 1"), plt.ylabel("Dimension 2")
    plt.savefig("meeting-3/tsne_plot_0_1")
    plt.show()

# Generate embeddings
clip_model.eval()
image_embeddings, image_labels = [], []
text_embeddings, text_labels = [], []

with torch.no_grad():
    for digit, data in [(0, x_test_0), (1, x_test_1)]:
        for i in range(50):
            img = torch.tensor(data[i], dtype=torch.float32).unsqueeze(0).to(device)
            for transform, shift in [
                (normal_transform, 'normal'),
                (left_shift_45_transform, 'left45'),
                (right_shift_45_transform, 'right45'),
                (left_shift_135_transform, 'left135'),
                (right_shift_135_transform, 'right135')
            ]:
                emb = clip_model.image_encoder(transform(img).unsqueeze(0).to(device))
                image_embeddings.append(emb.cpu().numpy().flatten())
                image_labels.append(f"{digit}_{shift}")
    
    for text in ["zero", "one"]:
        seq = torch.tensor(encode(text)).unsqueeze(0).to(device)
        emb = clip_model.text_encoder(seq)
        text_embeddings.append(emb.cpu().numpy().flatten())
        text_labels.append(text)

plot_tsne(np.array(image_embeddings), np.array(text_embeddings), image_labels, text_labels)

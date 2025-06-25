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
import matplotlib.colors as mcolors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Filter for 4s and 9s
x_train_4 = x_train[(y_train == 4)]
y_train_4 = y_train[(y_train == 4)]
x_train_9 = x_train[(y_train == 9)]
y_train_9 = y_train[(y_train == 9)]

x_test_4 = x_test[(y_test == 4)]
y_test_4 = y_test[(y_test == 4)]
x_test_9 = x_test[(y_test == 9)]
y_test_9 = y_test[(y_test == 9)]

x_train = np.concatenate([x_train_4, x_train_9], axis=0)
y_train = np.concatenate([y_train_4, y_train_9], axis=0)
x_test = np.concatenate([x_test_4, x_test_9], axis=0)
y_test = np.concatenate([y_test_4, y_test_9], axis=0)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

num_to_word = {
    4: 'four',
    9: 'nine'
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

batch_size = 64
train_dataset = MNISTDataset(x_train, y_train, text_labels_train) 
test_dataset = MNISTDataset(x_test, y_test, text_labels_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

chars = sorted(list(set("".join(text_labels_train))))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)
        # self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return nn.functional.normalize(x, dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # Average embeddings if sequence length > 1
        x = self.fc(x)
        return nn.functional.normalize(x, dim=-1)

class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(vocab_size)

    def forward(self, image, text):
        image_emb = self.image_encoder(image)
        text_emb = self.text_encoder(text)
        return image_emb, text_emb

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(ContrastiveLoss, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, image_emb, text_emb):
        logits = torch.matmul(image_emb, text_emb.T) / self.temperature
        labels = torch.arange(image_emb.size(0)).to(image_emb.device)
        return nn.CrossEntropyLoss()(logits, labels)

def calculate_accuracy(image_emb, text_emb):
    logits = torch.matmul(image_emb, text_emb.T)
    labels = torch.arange(image_emb.size(0)).to(image_emb.device)
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    return accuracy

clip_model = CLIP()
clip_model.to(device)

optimizer = optim.Adam(clip_model.parameters(), lr=0.001)
criterion = ContrastiveLoss()

epochs = 50
for epoch in range(epochs):
    clip_model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels, text_labels in train_loader:
        images = images.to(device)

        text_sequences = [torch.tensor(encode(label), dtype=torch.long) for label in text_labels]
        text_sequences = nn.utils.rnn.pad_sequence(text_sequences, batch_first=True).to(device)

        optimizer.zero_grad()

        # Forward pass
        image_emb, text_emb = clip_model(images, text_sequences)
        loss = criterion(image_emb, text_emb)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(image_emb, text_emb)
        running_acc += acc
        
    if epoch%10==0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {running_acc / len(train_loader):.4f}")

clip_model.eval()
test_loss = 0.0
test_acc = 0.0

with torch.no_grad():
    for images, labels, text_labels in test_loader:
        images = images.to(device)

        text_sequences = [torch.tensor(encode(label), dtype=torch.long) for label in text_labels]
        text_sequences = nn.utils.rnn.pad_sequence(text_sequences, batch_first=True).to(device)

        image_emb, text_emb = clip_model(images, text_sequences)
        loss = criterion(image_emb, text_emb)
        test_loss += loss.item()

        acc = calculate_accuracy(image_emb, text_emb)
        test_acc += acc

print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}")

def shift_image(image, shift_x):
    # Move the tensor to CPU before converting to NumPy
    image = image.cpu().squeeze().numpy() # Remove the batch and channel dimensions

    image = Image.fromarray((image * 255).astype(np.uint8()))

    width, height = image.size
    shifted_image = Image.new(image.mode, (width, height), color=0)

    for x in range(width):
        for y in range(height):
            new_x = (x + shift_x) % width  # Wrap around
            shifted_image.putpixel((new_x, y), image.getpixel((x, y)))

    shifted_image = np.array(shifted_image).astype(np.float32) / 255.0
    shifted_image = np.expand_dims(shifted_image, axis=0)  # Add batch dimension
    shifted_image = np.expand_dims(shifted_image, axis=1)  # Add channel dimension
    return torch.tensor(shifted_image, dtype=torch.float32)

def plot_tsne(embeddings, labels, title="t-SNE Plot"):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Create dictionaries for colors and markers
    color_map = {'4': 'blue', '9': 'red'}
    marker_map = {'right': 'o', 'left': 's'}  # 'o' for circle (right), 's' for square (left)

    plt.figure(figsize=(12, 8))

    # Plot each point
    for i, label in enumerate(labels):
        digit, direction = label.split('_')
        color = color_map[digit]
        marker = marker_map[direction]
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], c=color, marker=marker, s=50, alpha=0.7)

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='4 (right)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='4 (left)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='9 (right)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='9 (left)')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("trained-shift.png")
    plt.show()

# Prepare shifted images
clip_model.eval()
image_embeddings = []
labels = []

num_shifts = 50
shift_range = 5  # Shift by a few pixels

with torch.no_grad():
    # Process 4s
    for i in range(num_shifts):
        img = torch.tensor(x_test_4[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # The dimensions must be expanded correctly

        # Shift right
        shifted_img_right = shift_image(img, shift_range).to(device)
        emb_right = clip_model.image_encoder(shifted_img_right)
        image_embeddings.append(emb_right.cpu().numpy().flatten())
        labels.append("4_right")  # Use string labels for clarity

        # Shift left
        shifted_img_left = shift_image(img, -shift_range).to(device)
        emb_left = clip_model.image_encoder(shifted_img_left)
        image_embeddings.append(emb_left.cpu().numpy().flatten())
        labels.append("4_left")

    # Process 9s
    for i in range(num_shifts):
        img = torch.tensor(x_test_9[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # The dimensions must be expanded correctly

        # Shift right
        shifted_img_right = shift_image(img, shift_range).to(device)
        emb_right = clip_model.image_encoder(shifted_img_right)
        image_embeddings.append(emb_right.cpu().numpy().flatten())
        labels.append("9_right")

        # Shift left
        shifted_img_left = shift_image(img, -shift_range).to(device)
        emb_left = clip_model.image_encoder(shifted_img_left)
        image_embeddings.append(emb_left.cpu().numpy().flatten())
        labels.append("9_left")
image_embeddings = np.array(image_embeddings)
labels = np.array(labels)

plot_tsne(image_embeddings, labels, title="t-SNE for Shifted 4s and 9s")

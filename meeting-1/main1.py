import torch
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)
num_to_word = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'
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

# Data augmentation
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze(0).numpy()),  
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

batch_size = 64
train_dataset = MNISTDataset(x_train, y_train, text_labels_train, transform=transform)
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

epochs = 10
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

def plot_tsne(embeddings, name, labels, title="t-SNE Plot"):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(name)
    plt.show()

image_embeddings = []
text_embeddings = []
all_labels = []
clip_model.eval()

with torch.no_grad():
    for images, labels, text_labels in test_loader:
        images = images.to(device)

        text_sequences = [torch.tensor(encode(label), dtype=torch.long) for label in text_labels]
        text_sequences = nn.utils.rnn.pad_sequence(text_sequences, batch_first=True).to(device)

        img_emb = clip_model.image_encoder(images)
        txt_emb = clip_model.text_encoder(text_sequences)

        image_embeddings.append(img_emb.cpu())
        text_embeddings.append(txt_emb.cpu())
        all_labels.append(labels)

image_embeddings = torch.cat(image_embeddings, dim=0).numpy()
text_embeddings = torch.cat(text_embeddings, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

plot_tsne(image_embeddings, "meeting-1/tsne_image.png", all_labels, title="t-SNE for Image Embeddings")
plot_tsne(text_embeddings, "meeting-1/tsne_text.png", all_labels, title="t-SNE for Text Embeddings")
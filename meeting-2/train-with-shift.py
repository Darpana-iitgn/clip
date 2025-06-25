import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Filter 4s and 9s
train_mask = (y_train == 4) | (y_train == 9)
test_mask = (y_test == 4) | (y_test == 9)

x_train = x_train[train_mask]
y_train = y_train[train_mask]
x_test = x_test[test_mask]
y_test = y_test[test_mask]

# Create shifted versions
def create_shifted_dataset(images, labels, shift_amount=5):
    shifted_images = []
    new_labels = []
    text_labels = []
    
    for img, lbl in zip(images, labels):
        # Original
        shifted_images.append(img)
        new_labels.append(lbl)
        text_labels.append(f"{lbl}_normal")
        
        # Left shift
        shifted_images.append(np.roll(img, -shift_amount, axis=1))
        new_labels.append(lbl)
        text_labels.append(f"{lbl}_left")
        
        # Right shift
        shifted_images.append(np.roll(img, shift_amount, axis=1))
        new_labels.append(lbl)
        text_labels.append(f"{lbl}_right")
    
    return np.array(shifted_images), np.array(new_labels), text_labels

# Process training data
x_train = np.expand_dims(x_train, 1)  # Add channel dimension
x_train_shifted, y_train_shifted, text_train = create_shifted_dataset(x_train, y_train)

# Process test data (50% normal, 25% left, 25% right)
x_test = np.expand_dims(x_test, 1)  # Add channel dimension
x_test_final = []
y_test_final = []
text_test = []

for img, lbl in zip(x_test, y_test):
    # Add original
    x_test_final.append(img)
    y_test_final.append(lbl)
    text_test.append(f"{lbl}_normal")
    
    # Add left shift
    x_test_final.append(np.roll(img, -5, axis=2))
    y_test_final.append(lbl)
    text_test.append(f"{lbl}_left")
    
    # Add right shift
    x_test_final.append(np.roll(img, 5, axis=2))
    y_test_final.append(lbl)
    text_test.append(f"{lbl}_right")

x_test_final = np.array(x_test_final)
y_test_final = np.array(y_test_final)

# Custom Dataset class
class MNISTDataset(Dataset):
    def __init__(self, images, labels, text_labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.text_labels = text_labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create datasets
num_to_word = {4: "four", 9: "nine"}
train_dataset = MNISTDataset(x_train_shifted, y_train_shifted, text_train)
test_dataset = MNISTDataset(x_test_final, y_test_final, text_test)

# Model architecture (unchanged)
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64*7*7, 256)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=2, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 256)
        
    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        x = self.fc(hidden[-1])
        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.classifier = nn.Linear(512, 2)
        
    def forward(self, image, text):
        img_features = self.image_encoder(image)
        txt_features = self.text_encoder(text)
        combined = torch.cat([img_features, txt_features], dim=1)
        return self.classifier(combined)
    
# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = (labels == 9).long().to(device)  # Convert to binary: 4=0, 9=1
        text_input = (labels).unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images, text_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = (labels == 9).long().to(device)
        text_input = (labels).unsqueeze(1).to(device)
        outputs = model(images, text_input)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# T-SNE Visualization
def plot_tsne(embeddings, labels, title="t-SNE Plot of Test Set"):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    markers = {'normal': 'o', 'left': 's', 'right': '^'}
    colors = {'4': 'blue', '9': 'red'}
    
    for i, label in enumerate(labels):
        digit, direction = label.split('_')
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1],c=colors[digit], marker=markers[direction], alpha=0.6,edgecolors='w', s=50)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='4 Normal',markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='4 Left',markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='4 Right',markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='9 Normal',markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='9 Left',markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='9 Right',markerfacecolor='red', markersize=10)
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("train-with-shift.png")
    plt.show()

# Generate embeddings
all_embeddings = []
all_labels = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        embeddings = model.image_encoder(images)
        all_embeddings.append(embeddings.cpu().numpy())
all_embeddings = np.concatenate(all_embeddings)
plot_tsne(all_embeddings, text_test)

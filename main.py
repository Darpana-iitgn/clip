import torch
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

digit_to_text = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
text_labels = [digit_to_text[label] for label in y_train]

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

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return nn.functional.normalize(x, dim=-1)  

chars = sorted(list(set(text_labels)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
] 
decode = lambda l: "".join([itos[i] for i in l])

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=20):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.ff1 = nn.Linear(embed_dim, 32)
        self.ff2 = nn.Linear(32, 16)
        self.w_t = nn.Linear(16, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.ff1(x)
        x = self.ff2(x)
        x = self.w_t(x)
        return nn.functional.normalize(x, dim=-1)  

class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, image, text):
        image_emb = self.image_encoder(image)
        text_emb = self.text_encoder(text)

        image_emb = image_emb.to(image.device) 
        text_emb = text_emb.to(image.device)  
        
        return image_emb, text_emb

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_emb, text_emb):
        logits = torch.matmul(image_emb, text_emb.T) / self.temperature
        labels = torch.arange(image_emb.size(0)).to(image_emb.device)
        return nn.CrossEntropyLoss()(logits, labels)

train_images = torch.tensor(x_train, dtype=torch.float32)
train_labels = torch.tensor(y_train, dtype=torch.long)
test_images = torch.tensor(x_test, dtype=torch.float32)
test_labels = torch.tensor(y_test, dtype=torch.long)

clip_model = CLIP()
clip_model.to(device)
clip_model.train()

optimizer = optim.Adam(clip_model.parameters(), lr=0.001)
criterion = ContrastiveLoss()

epochs = 5  
batch_size = 64
num_batches = len(train_images) // batch_size

for epoch in range(epochs):
    clip_model.train()  
    running_loss = 0.0
    
    for i in range(num_batches):
        
        start = i * batch_size
        end = (i + 1) * batch_size
        images_batch = train_images[start:end]
        labels_batch = train_labels[start:end]
        
        optimizer.zero_grad()
        
        # Forward pass
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)
        image_emb, text_emb = clip_model(images_batch, labels_batch)

        # Backward pass and optimization
        loss = criterion(image_emb, text_emb)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/num_batches:.4f}")

clip_model.eval() 

with torch.no_grad():
    
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    test_image_embs, test_text_embs = clip_model(test_images, test_labels)

test_image_embs = test_image_embs.cpu().numpy()
test_labels = test_labels.cpu().numpy()

def plot_tsne(embeddings, labels, title="t-SNE Plot"):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("tsne.png")
    plt.show()

plot_tsne(test_image_embs, test_labels, title="t-SNE for Image Embeddings")

with torch.no_grad():
    text_embs = clip_model.text_encoder(torch.arange(10).to(device))
    text_embs = text_embs.cpu().numpy()

similarities = np.dot(test_image_embs, text_embs.T)
y_pred = np.argmax(similarities, axis=1)

accuracy = accuracy_score(test_labels, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
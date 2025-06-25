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

#normalization 
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


normal_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze().cpu().numpy()),  # Remove channel dimension
    transforms.ToPILImage(),  # Convert to PIL Image
    transforms.ToTensor()  # Convert back to tensor, adding channel dimension
])

left_shift_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze().cpu().numpy()),  # Remove channel dimension
    transforms.ToPILImage(),  # Convert to PIL Image
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=0, translate=(-5, 0), scale=1, shear=0)),  # Apply shift
    transforms.ToTensor()  # Convert back to tensor, adding channel dimension
])

right_shift_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.squeeze().cpu().numpy()),  # Remove channel dimension
    transforms.ToPILImage(),  # Convert to PIL Image
    transforms.Lambda(lambda img: transforms.functional.affine(img, angle=0, translate=(5, 0), scale=1, shear=0)),  # Apply shift
    transforms.ToTensor()  # Convert back to tensor, adding channel dimension
])

class CustomTestDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, text_label = self.dataset[idx]
        
        if idx < len(self.dataset) // 3:
            return left_shift_transform(image), label, text_label
        elif idx < 2 * len(self.dataset) // 3:
            return right_shift_transform(image), label, text_label
        else:
            return normal_transform(image), label, text_label


batch_size = 64
train_dataset = MNISTDataset(x_train, y_train, text_labels_train, transform=None)
test_dataset = MNISTDataset(x_test, y_test, text_labels_test)
custom_test_dataset = CustomTestDataset(test_dataset)
test_loader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=False)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=False)

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
# for epoch in range(epochs):
#     clip_model.train()
#     running_loss = 0.0
#     running_acc = 0.0

#     for images, labels, text_labels in train_loader:
#         images = images.to(device)

#         text_sequences = [torch.tensor(encode(label), dtype=torch.long) for label in text_labels]
#         text_sequences = nn.utils.rnn.pad_sequence(text_sequences, batch_first=True).to(device)

#         optimizer.zero_grad()

#         # Forward pass
#         image_emb, text_emb = clip_model(images, text_sequences)
#         loss = criterion(image_emb, text_emb)
#         running_loss += loss.item()

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         acc = calculate_accuracy(image_emb, text_emb)
#         running_acc += acc

#     print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {running_acc / len(train_loader):.4f}")

# clip_model.eval()
# test_loss = 0.0
# test_acc = 0.0

# with torch.no_grad():
#     for images, labels, text_labels in test_loader:
#         images = images.to(device)

#         text_sequences = [torch.tensor(encode(label), dtype=torch.long) for label in text_labels]
#         text_sequences = nn.utils.rnn.pad_sequence(text_sequences, batch_first=True).to(device)

#         image_emb, text_emb = clip_model(images, text_sequences)
#         loss = criterion(image_emb, text_emb)
#         test_loss += loss.item()

#         acc = calculate_accuracy(image_emb, text_emb)
#         test_acc += acc

# print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}")


def plot_tsne(image_embeddings, text_embeddings, image_labels, text_labels, title="t-SNE Plot"):
    # Combine image and text embeddings
    all_embeddings = np.vstack((image_embeddings, text_embeddings))
    all_labels = np.concatenate((image_labels, text_labels))

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    # Split back into image and text embeddings
    reduced_image_embeddings = reduced_embeddings[:len(image_embeddings)]
    reduced_text_embeddings = reduced_embeddings[len(image_embeddings):]

    plt.figure(figsize=(12, 8))

    # Plot image embeddings
    colors = {'4': 'blue', '9': 'red'}
    markers = {'normal': 'o', 'left': 's', 'right': '^'}
    for i, label in enumerate(image_labels):
        digit, shift = label.split('_')
        if digit == '4':
            if shift == 'normal':
                color = 'blue'
            elif shift == 'left':
                color = 'red'
            else:  # right
                color = 'green'
        else:  # '9'
            if shift == 'normal':
                color = 'pink'
            elif shift == 'left':
                color = 'yellow'
            else:  # right
                color = 'magenta'
        plt.scatter(reduced_image_embeddings[i, 0], reduced_image_embeddings[i, 1], 
                    c=color, marker=markers[shift], s=50, alpha=0.7)

    # Plot text embeddings
    text_colors = {'four': 'cyan', 'nine': 'magenta'}
    for i, label in enumerate(text_labels):
        plt.scatter(reduced_text_embeddings[i, 0], reduced_text_embeddings[i, 1], 
                    c=text_colors[label], marker='*', s=200, alpha=1)

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='4 (normal)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='4 (left)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='4 (right)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='9 (normal)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=10, label='9 (left)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='magenta', markersize=10, label='9 (right)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan', markersize=15, label='"four"'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', markersize=15, label='"nine"')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("meeting-3/tsne_plot.png")
    plt.show()

# # Prepare data
# clip_model.eval()
# image_embeddings = []
# image_labels = []
# text_embeddings = []
# text_labels = []

# num_samples = 50  # Number of samples per category

# with torch.no_grad():
#     # Process images
#     for digit, x_test in [(4, x_test_4), (9, x_test_9)]:
#         for i in range(num_samples):
#             img = torch.tensor(x_test[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
#             # Normal
#             emb_normal = clip_model.image_encoder(normal_transform(img).unsqueeze(0).to(device))
#             image_embeddings.append(emb_normal.cpu().numpy().flatten())
#             image_labels.append(f"{digit}_normal")
            
#             # Left shift
#             emb_left = clip_model.image_encoder(left_shift_transform(img).unsqueeze(0).to(device))
#             image_embeddings.append(emb_left.cpu().numpy().flatten())
#             image_labels.append(f"{digit}_left")
            
#             # Right shift
#             emb_right = clip_model.image_encoder(right_shift_transform(img).unsqueeze(0).to(device))
#             image_embeddings.append(emb_right.cpu().numpy().flatten())
#             image_labels.append(f"{digit}_right")
    
#     # Process text
#     for text in ["four", "nine"]:
#         text_sequence = torch.tensor(encode(text), dtype=torch.long).unsqueeze(0).to(device)
#         text_emb = clip_model.text_encoder(text_sequence)
#         text_embeddings.append(text_emb.cpu().numpy().flatten())
#         text_labels.append(text)

# image_embeddings = np.array(image_embeddings)
# text_embeddings = np.array(text_embeddings)

# Plot
# plot_tsne(image_embeddings, text_embeddings, image_labels, text_labels, title="t-SNE for Image and Text Embeddings")

import matplotlib.pyplot as plt
import torch

def show_and_save_augmented_grid(x_test_4, x_test_9, normal_transform, left_shift_transform, right_shift_transform, device):
    num_samples = 3
    digits = [(4, x_test_4), (9, x_test_9)]
    augmentations = [
        ("Normal", normal_transform),
        ("Left Shift", left_shift_transform),
        ("Right Shift", right_shift_transform),
    ]

    fig, axes = plt.subplots(2, 9, figsize=(18,4))

    for row_idx, (digit, x_test_digit) in enumerate(digits):
        for aug_idx, (aug_name, aug_transform) in enumerate(augmentations):
            for i in range(num_samples):
                img = torch.tensor(x_test_digit[i], dtype=torch.float32).unsqueeze(0)
                img_aug = aug_transform(img).squeeze().cpu().numpy()
                ax = axes[row_idx, aug_idx * num_samples + i]
                ax.imshow(img_aug, cmap='gray')
                ax.axis('off')
                if row_idx == 0:
                    ax.set_title(f"{aug_name}\n{i+1}", fontsize=10)
            axes[row_idx, 0].set_ylabel(f"Digit {digit}", fontsize=12)

    plt.tight_layout()
    plt.savefig("augmented_images_grid.png", bbox_inches='tight')
    plt.show()

# Call the function with your actual data and transforms
show_and_save_augmented_grid(x_test_4, x_test_9, normal_transform, left_shift_transform, right_shift_transform, device)

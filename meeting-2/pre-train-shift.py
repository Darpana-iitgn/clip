import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set random seeds 
torch.manual_seed(42)
np.random.seed(42)

def load_mnist(batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1
    ])

    dataset_root = "/home/project/IntCLIP/clip/mnist_data"

    test_dataset = datasets.MNIST(root=dataset_root, train=False, download=False, transform=transform)
    
    # Filter only digits 4 and 9
    idx = (test_dataset.targets == 4) | (test_dataset.targets == 9)
    test_dataset.data = test_dataset.data[idx]
    test_dataset.targets = test_dataset.targets[idx]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

class ShiftedMNISTDataset(Dataset):
    def __init__(self, dataset, shift_pixels=2, max_samples_per_class=50):
        self.shift_pixels = shift_pixels
        self.data = []
        
        count_4, count_9 = 0, 0
        for image, label in dataset:
            image_pil = transforms.ToPILImage()(image)
            image_size = image_pil.size  # Ensure (width, height) format

            if label == 4 and count_4 < max_samples_per_class:
                self.data.append((transforms.ToTensor()(image_pil.transform(image_size, Image.AFFINE, (1, 0, -self.shift_pixels, 0, 1, 0))), "Left Shifted 4"))
                self.data.append((transforms.ToTensor()(image_pil.transform(image_size, Image.AFFINE, (1, 0, self.shift_pixels, 0, 1, 0))), "Right Shifted 4"))
                count_4 += 1
            elif label == 9 and count_9 < max_samples_per_class:
                self.data.append((transforms.ToTensor()(image_pil.transform(image_size, Image.AFFINE, (1, 0, -self.shift_pixels, 0, 1, 0))), "Left Shifted 9"))
                self.data.append((transforms.ToTensor()(image_pil.transform(image_size, Image.AFFINE, (1, 0, self.shift_pixels, 0, 1, 0))), "Right Shifted 9"))
                count_9 += 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class CustomClassifier:
    def __init__(self, prompts):
        self.class_prompts = prompts
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load('RN50', device=self.device)
        self.preprocessed_text = clip.tokenize(self.class_prompts).to(self.device)
        print(f'Class Prompts: {self.class_prompts}')
    
    def classify(self, image, y_true=None):
        preprocessed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_logits, _ = self.model(preprocessed_image, self.preprocessed_text)
            proba_list = image_logits.softmax(dim=-1).cpu().numpy()[0]
        
        y_pred = np.argmax(proba_list)
        y_pred_proba = np.max(proba_list)
        y_pred_token = self.class_prompts[y_pred]
        return y_true, y_pred, y_pred_token, y_pred_proba

def extract_embeddings(model, data_loader, device):
    model.eval()  
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, lbls in data_loader:
            preprocessed_images = torch.stack([classifier.preprocess(transforms.ToPILImage()(img)) for img in images]).to(device)
            output = model.encode_image(preprocessed_images)
            embeddings.append(output.cpu().numpy())
            labels.extend(lbls)
    
    embeddings = np.vstack(embeddings)  
    return embeddings, labels

def plot_tsne(embeddings, labels, name, title="t-SNE Plot"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))  # Ensure order is consistent
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_map[label] for label in labels]

    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=numeric_labels, cmap='tab10', s=15)
    
    # Set colorbar with proper labels
    colorbar = plt.colorbar(scatter)
    colorbar.set_ticks(range(len(unique_labels)))
    colorbar.set_ticklabels(unique_labels)

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(name)
    plt.show()

def test_classifier(classifier, data_loader):
    y_true_list = []
    y_pred_list = []

    for images, labels in tqdm(data_loader, desc="Testing"):
        for img, label in zip(images, labels):
            y_true, y_pred, y_pred_token, _ = classifier.classify(transforms.ToPILImage()(img), label)
            y_true_list.append(y_true)
            y_pred_list.append(y_pred_token)
    
    accuracy = accuracy_score(y_true_list, y_pred_list)
    print(f"Classification Accuracy on Shifted Images: {accuracy * 100:.2f}%")
    return accuracy

# Main Execution
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader = load_mnist()

    # Create shifted dataset
    shifted_dataset = ShiftedMNISTDataset(test_loader.dataset)
    shifted_loader = DataLoader(shifted_dataset, batch_size=32, shuffle=False)

    prompts = ["Left Shifted 9", "Right Shifted 9", "Left Shifted 4", "Right Shifted 4"]
    classifier = CustomClassifier(prompts)

    # Extract embeddings for shifted images
    embeddings, labels = extract_embeddings(classifier.model, shifted_loader, device)

    # Plot t-SNE of shifted images
    plot_tsne(embeddings, labels, "pre_train_shift.png", title="t-SNE of Shifted Images (Left and Right 4s and 9s)")

    # Test classifier and print accuracy
    test_accuracy = test_classifier(classifier, shifted_loader)

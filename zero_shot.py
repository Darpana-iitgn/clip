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

    train_dataset = datasets.MNIST(root=dataset_root, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=dataset_root, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class MNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

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

    def validate(self, dataloader):
        df_results = []
        y_true_list, y_pred_list = [], []
        
        for images, labels in tqdm(dataloader):
            for i in range(len(images)):
                image = transforms.ToPILImage()(images[i])  
                y_true, y_pred, y_pred_token, proba = self.classify(image, labels[i].item())
                df_results.append({'y_true': y_true, 'y_pred': y_pred, 'y_pred_token': y_pred_token, 'proba': proba})
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
        
        accuracy = accuracy_score(y_true_list, y_pred_list)
        return df_results, accuracy

def extract_embeddings(model, data_loader, device):
    model.eval()  
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, lbls in data_loader:
            
            preprocessed_images = torch.stack([classifier.preprocess(transforms.ToPILImage()(img)) for img in images]).to(device)

            output = model.encode_image(preprocessed_images)
            embeddings.append(output.cpu().numpy())
            labels.append(lbls.numpy())
    
    embeddings = np.vstack(embeddings)  
    labels = np.hstack(labels)          
    return embeddings, labels

def plot_tsne(embeddings, labels, name, title="t-SNE Plot"):
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

# Main Execution
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader = load_mnist()

    prompts = [f"An image of the digit {i}" for i in range(10)]

    classifier = CustomClassifier(prompts)

    results, accuracy = classifier.validate(test_loader)
    print(f"Test Accuracy: {accuracy}")

    embeddings, labels = extract_embeddings(classifier.model, test_loader, device)

    plot_tsne(embeddings, labels, "zero_shot.png", title="t-SNE of zero-shot CLIP on MNIST")
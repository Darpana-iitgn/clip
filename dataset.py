import os
from torch.utils.data import Dataset
from PIL import Image

class RealEstate10KDataset(Dataset):
    def __init__(self, image_dir, captions_file, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.samples = []
        with open(captions_file, 'r') as f:
            for line in f:
                img_name, caption = line.strip().split('\t')
                self.samples.append((img_name, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}
    


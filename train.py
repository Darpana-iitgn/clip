import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from lora import LoraConfig, LoraModel
from dataset import RealEstate10KDataset
import os

def main():
    # Paths
    image_dir = "data/images"
    captions_file = "data/captions.txt"
    model_save_dir = "models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Load processor and dataset
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = RealEstate10KDataset(image_dir, captions_file, processor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Load CLIP and wrap with LoRA
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    lora_config = LoraConfig(
        target_modules=["visual.proj", "text_projection", "q_proj", "k_proj", "v_proj"],  # Adjust as needed
        rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        use_rslora=True,
        bias="lora_only"
    )
    lora_clip = LoraModel(clip_model, lora_config)
    lora_clip.to("cuda")
    lora_clip.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, lora_clip.parameters()), lr=3e-4)

    # Training loop
    for epoch in range(5):
        for batch in dataloader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            pixel_values = batch['pixel_values'].cuda()

            outputs = lora_clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Example loss: cosine similarity (replace with geometric loss as needed)
            loss = 1 - torch.cosine_similarity(image_embeds, text_embeds).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

    # Save LoRA weights only
    lora_clip.save_model(os.path.join(model_save_dir, "lora_clip_geometric.safetensors"), merge_weights=False)

if __name__ == "__main__":
    main()

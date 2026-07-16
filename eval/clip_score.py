import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
import re


class CLIP_Score():
    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda:2' if torch.cuda.is_available() else 'cpu'):
        self.model = CLIPModel.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version, use_fast=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.device = device
        self.model = self.model.to(self.device)
    
    def __call__(self, dataloader):
        out_score = 0
        for item in dataloader:
            image_path = item['image'][0]
            caption = item['text'][0]
            image = [Image.open(image_path).convert("RGB")]
            out_score_matrix = self.model_output([caption], image)
            out_score += out_score_matrix.mean().item() 
        return out_score / len(dataloader)
    
    def model_output(self, text, img):
        torch.cuda.empty_cache()
        image_inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        image_feats = self.model.get_image_features(**image_inputs)

        text_inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        text_feats = self.model.get_text_features(**text_inputs)

        image_feats = image_feats / image_feats.norm(dim=1, p=2, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=1, p=2, keepdim=True)
        score = (image_feats * text_feats).sum(-1)
        return score




class ImageCaptionDataset(Dataset):
    def __init__(self, folder_path):
        self.image_paths = []
        self.captions = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                          
                if 'dog' not in file:
                    continue
                self.image_paths.append(os.path.join(folder_path, file))
                caption = os.path.splitext(file)[0]
                caption = re.sub(r'[_-]\d+$', '', caption)
                caption = caption.replace('_', ' ')
                self.captions.append(caption)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return {
            "image": self.image_paths[idx],  
            "text": self.captions[idx]
        }



if __name__ == "__main__":
    folder = "/outputs/erase_dog/dog"  
    print(folder)
    dataset = ImageCaptionDataset(folder)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    clip_scorer = CLIP_Score()
    avg_score = clip_scorer(dataloader)

    print(f"📊 Average CLIPScore for folder '{folder}': {avg_score:.4f}")

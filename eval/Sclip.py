import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class SCLIPScore:
    def __init__(self, model_name='openai/clip-vit-large-patch14', device=None):
        """
        Compute CLIP cosine similarity score between images and a text style prompt.
        """
        self.device = device or ('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def score(self, image_paths, style_prompt):
        """
        image_paths: list of image file paths
        style_prompt: text prompt describing the target style
        """
        scores = []

        # Encode text once
        text_inputs = self.processor(
            text=[style_prompt],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Skipping image {repr(img_path)} due to load error: {repr(e)}")
                continue

            image_inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            score = (image_features * text_features).sum().item()
            scores.append(score)

        if len(scores) == 0:
            print("⚠️ No valid images found.")
            return 0.0

        avg_score = sum(scores) / len(scores)
        print(f"✔ Average SCLIP score: {avg_score:.4f}")
        return avg_score


# ================= Example Usage =================
if __name__ == "__main__":
    folder = "/outputs/erase_Picasso/Picasso"
    style = "Picasso"
    style_prompt = f"in the style of {style}"

    # Collect image paths whose filenames contain the style keyword
    image_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        and style.lower() in f.lower()  # filename must contain the keyword
    ]

    if not image_paths:
        print(f"⚠️ No images found containing keyword '{style}'.")
    else:
        scorer = SCLIPScore()
        avg_score = scorer.score(image_paths, style_prompt)

import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class CLIPConceptCounter:
    def __init__(self, concept="dog", model_name="openai/clip-vit-large-patch14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.concept_prompt = f"a photo of a {concept}"
        self.other_prompt = "a photo of something else"

    @torch.no_grad()
    def is_concept(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image {image_path}: {e}")
            return False

        # Process image and text independently
        image_inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        text_inputs = self.processor(
            text=[self.concept_prompt, self.other_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Extract features
        image_features = self.model.get_image_features(**image_inputs)
        text_features = self.model.get_text_features(**text_inputs)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = image_features @ text_features.T
        pred = similarity.argmax().item()
        return pred == 0  # 0 = matches the concept prompt

    def count_concept_in_folder(self, folder_path):
        """Count images belonging to the target concept in a folder."""
        total_count = 0
        concept_count = 0

        for file in os.listdir(folder_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue
            # if "dog" not in file:
            #     continue
            total_count += 1
            image_path = os.path.join(folder_path, file)

            if self.is_concept(image_path):
                concept_count += 1
                print(f"✔ {file} → matches concept: '{self.concept_prompt}'")
            else:
                print(f"✘ {file} → does NOT match concept")

        if total_count == 0:
            print("No images found in folder.")
            return 0.0

        ratio = concept_count / total_count
        print(
            f"\nTotal images: {total_count}, "
            f"Matched concept images: {concept_count}, "
            f"Ratio: {ratio:.4f}"
        )
        return ratio


# ================= Usage Example =================
if __name__ == "__main__":
    folder = "/outputs/erase_dog/dog"
    counter = CLIPConceptCounter(concept="dog")
    ratio = counter.count_concept_in_folder(folder)

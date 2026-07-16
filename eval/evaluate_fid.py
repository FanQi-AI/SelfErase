import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import linalg

# ================= Dataset =================
class ImageFolderDataset(Dataset):
    def __init__(self, root_folder, concept=None, transform=None, min_count=None):
        """
        Load images from a folder. Optionally skip images containing a concept keyword in filename.
        If min_count is specified, image list will be repeated to reach the minimum count.
        """
        self.files = []
        for r, _, fnames in os.walk(root_folder):
            for f in fnames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    # Skip images containing the target concept keyword in filename
                    if concept and concept.lower() in f.lower():
                        continue
                    self.files.append(os.path.join(r, f))

        # Repeat samples if needed
        if min_count and len(self.files) > 0:
            repeat_times = int(np.ceil(min_count / len(self.files)))
            self.files = (self.files * repeat_times)[:min_count]

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            # Safe logging, no personal information
            print(f"⚠️ Skipping corrupted image {repr(img_path)}: {repr(e)}")
            # return a placeholder tensor to avoid DataLoader crash
            img = torch.zeros(3, 299, 299)
        return img


# ================= FID Calculator =================
class StableFIDCalculator:
    def __init__(self, device=None, batch_size=32, eps=1e-6, num_workers=4):
        """
        Computes FID using InceptionV3 from torchvision.
        """
        self.device = device or ("cuda:2" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.eps = eps
        self.num_workers = num_workers

        # Load torchvision InceptionV3
        self.model = models.inception_v3(pretrained=True, transform_input=True).to(self.device)
        self.model.eval()
        self.model.fc = torch.nn.Identity()  # Remove classification head

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def _get_features(self, folder, concept=None, min_count=None):
        dataset = ImageFolderDataset(folder, concept=concept, transform=self.transform, min_count=min_count)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        all_feats = []
        for batch in tqdm(loader, desc=f"Processing {os.path.basename(folder)}"):
            batch = batch.to(self.device)
            feats = self.model(batch)
            all_feats.append(feats.cpu().numpy())

        if not all_feats:
            return np.zeros((0, 2048))
        return np.concatenate(all_feats, axis=0)

    @staticmethod
    def _calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Standard FID computation with numerical stabilization.
        """
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return max(fid, 0.0)

    def compute_fid(self, folder1, folder2, concept=None, min_count=5000):
        feats1 = self._get_features(folder1, concept=concept, min_count=min_count)
        feats2 = self._get_features(folder2, concept=concept, min_count=min_count)

        if feats1.shape[0] == 0 or feats2.shape[0] == 0:
            print("⚠️ No valid images found. FID cannot be computed.")
            return None

        mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
        mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

        fid_value = self._calculate_fid(mu1, sigma1, mu2, sigma2, eps=self.eps)
        print(f"\n✔ FID = {fid_value:.4f}")
        return fid_value


# ================= Usage Example =================
if __name__ == "__main__":
    dir_real = "/outputs/10/erase_dog"
    dir_fake = "/outputs/oral_erase_dog"
    concept = "dog"  # skip images containing this substring

    fid_calc = StableFIDCalculator(batch_size=64, num_workers=8)
    fid = fid_calc.compute_fid(dir_real, dir_fake, concept=concept, min_count=200)

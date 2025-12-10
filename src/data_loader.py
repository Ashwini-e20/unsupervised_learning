# src/data_loader.py

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import IMAGE_SIZE, DATA_DIR, BATCH_SIZE, NUM_WORKERS

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir=DATA_DIR):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225],   # ImageNet stds
            ),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_path


def create_dataloader():
    dataset = UnlabeledImageDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    return dataloader

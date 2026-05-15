import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


LABEL_MAP = {
    "normal": 0,
    "turbulent": 1,
    "asymmetric": 2,
    "blocked": 3,
}

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class AirflowImageDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.transform = TRAIN_TRANSFORMS if split == "train" else EVAL_TRANSFORMS
        self.samples = []
        for label_name, label_id in LABEL_MAP.items():
            folder = os.path.join(root, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, fname), label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)


def get_loaders(root: str, batch_size: int = 32, num_workers: int = 4):
    train_ds = AirflowImageDataset(root, split="train")
    val_ds   = AirflowImageDataset(root, split="val")
    persist  = num_workers > 0
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=persist)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=persist)
    return train_dl, val_dl

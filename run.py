import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.io import read_image

# -----------------------------
# Dataset Loader
# -----------------------------
class WeatherDataset(Dataset):
    def __init__(self, folder: Path):
        self.paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.labels = [self._extract_label(p.name) for p in self.paths]
        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def _extract_label(self, filename: str) -> str:
        for i, ch in enumerate(filename):
            if ch.isdigit():
                return filename[:i]
        return filename.split(".")[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        try:
            img = read_image(path).float() / 255.0
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:
                img = img[:3, :, :]
            elif img.shape[0] != 3:
                return self.__getitem__((idx + 1) % len(self.paths))
            img = TF.resize(img, [128, 128], antialias=True)
            if img.shape != (3, 128, 128):
                return self.__getitem__((idx + 1) % len(self.paths))
        except Exception:
            return self.__getitem__((idx + 1) % len(self.paths))
        label = self.class_to_idx[self.labels[idx]]
        return img, label

# -----------------------------
# CNN Model (<5M parameters)
# -----------------------------
class MediumCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# -----------------------------
# Train Loop
# -----------------------------
def train(model, loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        train_acc = (correct / total) * 100
        # Validation after each epoch
        val_acc = evaluate(model, val_loader, device, silent=True)
    print(f"Epoch {epoch+1}: Loss={running_loss:.4f}  Train Acc={train_acc:.2f}%  Val Acc={val_acc:.2f}%")

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device, silent=False):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    acc = (correct / total) * 100
    if not silent:
        print(f"Final Test Accuracy: {acc:.2f}%")
    return acc

# -----------------------------
# Main
# -----------------------------
import random

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train and test weather classifier.")
    parser.add_argument("test_folder", nargs="?", default=None, help="Path to external test folder.")
    args = parser.parse_args()

    # Load training data
    trainval_dataset = WeatherDataset(Path("Weather"))
    n = len(trainval_dataset)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_split = int(0.8 * n)
    val_split = int(0.9 * n)
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]
    from torch.utils.data import Subset
    train_dataset = Subset(trainval_dataset, train_idx)
    val_dataset = Subset(trainval_dataset, val_idx)
    internal_test_dataset = Subset(trainval_dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)
    internal_test_loader = DataLoader(internal_test_dataset, batch_size=32, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MediumCNN(num_classes=len(trainval_dataset.classes)).to(device)

    print("Training...")
    train(model, train_loader, val_loader, device)

    # If external test folder is provided, use it
    if args.test_folder:
        print(f"Testing with external folder: {args.test_folder}")
        ext_test_dataset = WeatherDataset(Path(args.test_folder))
        ext_test_loader = DataLoader(ext_test_dataset, batch_size=32, num_workers=2)
        evaluate(model, ext_test_loader, device)
    else:
        print("Testing with internal test split...")
        evaluate(model, internal_test_loader, device)

if __name__ == "__main__":
    main()

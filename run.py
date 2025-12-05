import sys
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.models import resnet18

# ============================================================
# Dataset
# ============================================================
class WeatherDataset(Dataset):
    def __init__(self, folder: Path, transform=None):
        self.transform = transform

        # Accept common image types
        self.paths = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )

        if len(self.paths) == 0:
            raise ValueError(f"No images found in: {folder}")

        # Extract labels from filenames (prefix before any digit)
        self.labels = [self._extract_label(p.name) for p in self.paths]
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def _extract_label(self, filename: str):
        for i, c in enumerate(filename):
            if c.isdigit():
                return filename[:i]
        return filename.split(".")[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            img = read_image(str(img_path)).float() / 255.0
        except Exception:
            # Skip corrupted/unsupported images
            return self.__getitem__((idx + 1) % len(self.paths))

        # normalize channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]

        label = self.class_to_idx[self.labels[idx]]

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# Medium CNN (your custom model)
# ============================================================
class MediumCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# Training Loop (Improved)
# ============================================================
def train(model, train_loader, val_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        train_acc = 100 * correct / total
        # Validation
        val_acc, val_loss = evaluate(model, val_loader, device, return_loss=True)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Loss: {running_loss:.4f}"
        )


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, loader, device, return_loss=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total, correct = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total_loss += criterion(outputs, labels).item()

    acc = 100 * correct / total

    if return_loss:
        return acc, total_loss
    return acc


# ============================================================
# Main
# ============================================================
def main():

    weather_dir = Path("Weather")
    dataset = WeatherDataset(
        weather_dir,
        transform=T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ConvertImageDtype(torch.float32),
        ]),
    )

    # Split dataset
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)

    train_idx = idx[:int(0.75 * n)]
    val_idx   = idx[int(0.75 * n):int(0.9 * n)]
    test_idx  = idx[int(0.9 * n):]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=32)
    test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=32)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    num_classes = len(dataset.classes)

    # Choose model
    model = MediumCNN(num_classes).to(device)
    # OR use better pretrained architecture:
    # model = resnet18(weights=None)
    # model.fc = nn.Linear(512, num_classes)
    # model.to(device)

    train(model, train_loader, val_loader, device, epochs=20)
    print("\nEvaluating on test set...")
    test_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()

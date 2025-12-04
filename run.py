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
        self.paths = sorted(
            [
                p
                for p in folder.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
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
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# resnet18 blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# -----------------------------
# Train Loop
# -----------------------------
def train(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
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
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        train_acc = (correct / total) * 100

        # val
        model.eval()
        val_total, val_correct = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()
        val_acc = (val_correct / val_total) * 100

        print(
            f"Epoch {epoch+1}: Loss={running_loss:.4f}  Train Acc={train_acc:.2f}% Val Loss={val_loss:.4f} Val Acc={val_acc:.2f}%"
        )


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):
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
    print(f"Final Test Accuracy: {acc:.2f}%")


# -----------------------------
# Main
# -----------------------------
import random


def main():
    # dataset: Weather/
    all_dataset = WeatherDataset(Path("Weather"))
    n = len(all_dataset)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.75 * n)
    train_idx, val_idx, test_idx = (
        indices[:split],
        indices[split : int(0.9 * n)],
        indices[int(0.9 * n) :],
    )
    from torch.utils.data import Subset

    train_dataset = Subset(all_dataset, train_idx)
    val_dataset = Subset(all_dataset, val_idx)
    test_dataset = Subset(all_dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Number of classes: {len(all_dataset.classes)}")
    print(f"Classes: {all_dataset.classes}")
    # model = ResNet18(num_classes=len(all_dataset.classes)).to(device)
    model = MediumCNN(num_classes=len(all_dataset.classes)).to(device)
    print("Training...")
    train(model, train_loader, val_loader, device)
    print("Evaluating...")
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()

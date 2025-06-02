import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE   = 128
EPOCHS       = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
CUTOUT_SIZE  = 8
CUTOUT_PROB  = 0.5


class RandomCutout(object):

    def __init__(self, size: int, p: float = 0.5):
        self.size = size
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        if random.random() > self.p:
            return img

        _, h, w = img.shape

        y_center = np.random.randint(0, h)
        x_center = np.random.randint(0, w)

        half = self.size // 2
        y1 = np.clip(y_center - half, 0, h)
        y2 = np.clip(y_center + half, 0, h)
        x1 = np.clip(x_center - half, 0, w)
        x2 = np.clip(x_center + half, 0, w)

        img[:, y1:y2, x1:x2] = 0.0
        return img

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.fc(x)
        return x


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc  = 100.0 * running_correct / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            running_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc  = 100.0 * running_correct / total_samples
    return epoch_loss, epoch_acc


if __name__ == "__main__":

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RandomCutout(size=CUTOUT_SIZE, p=CUTOUT_PROB),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2430, 0.2610))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2430, 0.2610)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform_train
    )
    test_dataset  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model      = SimpleCNN(num_classes=10).to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_losses, train_accuracies = [], []
    test_losses,  test_accuracies  = [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoka {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:5.2f}% | "
              f" Test Loss: {test_loss:.4f}, Acc: {test_acc:5.2f}%")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss per Epoch (Cutout)')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy per Epoch (Cutout)')
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves_cutout.png')
    plt.show()

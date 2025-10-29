import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import SpheroidDataset
from pretrained_model import UNetWithResnet34Encoder

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
EPOCHS = 50
IMG_HEIGHT = 256
IMG_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directories
IMAGE_DIR = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/processed/images"
MASK_DIR = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/Segmentations/channels_cellpose"
MODEL_SAVE_PATH = "/content/drive/MyDrive/DDLS-Course/FinalProject/step3_unet_model/unet_spheroid.pth"

# Transformations
train_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

val_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
])

# Dataset and Dataloader
def main():
    dataset = SpheroidDataset(IMAGE_DIR, MASK_DIR)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size # Ensure all samples are covered
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform # Use validation transform for test set

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetWithResnet34Encoder(n_classes=3).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE, dtype=torch.float)
            masks = masks.to(DEVICE, dtype=torch.float)

            # Forward pass
            outputs = model(images)
            bce_loss = bce_loss_fn(outputs, masks)
            dice_loss = dice_loss_fn(outputs, masks)
            loss = 0.5 * bce_loss + 0.5 * dice_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_score(outputs, masks)

        train_loss = running_loss / len(train_loader)
        train_dice = running_dice / len(train_loader)
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)

        # Validation loop
        model.eval()
        running_loss = 0.0
        running_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, dtype=torch.float)
                masks = masks.to(DEVICE, dtype=torch.float)
                outputs = model(images)
                bce_loss = bce_loss_fn(outputs, masks)
                dice_loss = dice_loss_fn(outputs, masks)
                loss = 0.5 * bce_loss + 0.5 * dice_loss
                running_loss += loss.item()
                running_dice += dice_score(outputs, masks)

        val_loss = running_loss / len(val_loader)
        val_dice = running_dice / len(val_loader)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Test loop
    model.eval()
    test_loss = 0.0
    test_dice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE, dtype=torch.float)
            masks = masks.to(DEVICE, dtype=torch.float)
            outputs = model(images)
            bce_loss = bce_loss_fn(outputs, masks)
            dice_loss = dice_loss_fn(outputs, masks)
            loss = 0.5 * bce_loss + 0.5 * dice_loss
            test_loss += loss.item()
            test_dice += dice_score(outputs, masks)

    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")

    # Plotting
    plot_history(history)

def dice_score(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().cpu()

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig("/content/drive/MyDrive/DDLS-Course/FinalProject/step3_unet_model/loss_plot.png")

    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Dice Score')
    plt.legend()
    plt.savefig("/content/drive/MyDrive/DDLS-Course/FinalProject/step3_unet_model/dice_plot.png")

if __name__ == '__main__':
    main()
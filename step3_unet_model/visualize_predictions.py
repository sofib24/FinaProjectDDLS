import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import SpheroidDataset
from pretrained_model import UNetWithResnet34Encoder

# Parameters
MODEL_PATH = "/content/drive/MyDrive/DDLS-Course/FinalProject/step3_unet_model/unet_spheroid.pth"
IMAGE_DIR = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/processed/images"
MASK_DIR = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/Segmentations/channels_cellpose"
IMG_HEIGHT = 256
IMG_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EXAMPLES = 10

# Transformations
val_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    ToTensorV2(),
])

def main():
    # Load dataset and dataloader
    dataset = SpheroidDataset(IMAGE_DIR, MASK_DIR, transform=val_transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Load model
    model = UNetWithResnet34Encoder(n_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # Visualize predictions
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            if i >= NUM_EXAMPLES:
                break

            image = image.to(DEVICE, dtype=torch.float)
            pred_mask = model(image)
            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = (pred_mask > 0.1).float().cpu().numpy() # Lower threshold to 0.1

            image = image.cpu().numpy()

            # Plot
            fig, axes = plt.subplots(1, 4, figsize=(20, 5)) # 1 row, 4 columns

            # Original Image
            axes[0].imshow(image[0, 0], cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Predicted Masks
            # Use cmap='gray' and ensure values are 0 or 1 for binary masks
            axes[1].imshow(pred_mask[0, 0], cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Pred Endothelial')
            axes[1].axis('off')

            axes[2].imshow(pred_mask[0, 1], cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Pred Tumor')
            axes[2].axis('off')

            axes[3].imshow(pred_mask[0, 2], cmap='gray', vmin=0, vmax=1)
            axes[3].set_title('Pred Fibroblasts')
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(f"/content/drive/MyDrive/DDLS-Course/FinalProject/step3_unet_model/prediction_visualization_{i}.png")
            plt.close()

if __name__ == '__main__':
    main()
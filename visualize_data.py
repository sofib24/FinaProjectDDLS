
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile
import os
from torchvision import transforms, models
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from PIL import Image

# Suppress warnings from tifffile
warnings.filterwarnings("ignore", category=UserWarning, module='tifffile')

class SpheroidSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        # Channel names corresponding to Endothelial, Tumor, Fibroblasts
        self.class_names = ['Endothelial', 'Tumor', 'Fibroblasts']
        self.label_channels = ['ch002', 'ch003', 'ch004']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)

        # Load the grayscale image
        image = tifffile.imread(img_path)
        image = Image.fromarray(image)

        # Create the label mask
        mask = []
        base_name = img_name.replace('.tif', '')
        
        for ch in self.label_channels:
            mask_file_name = f"{base_name}_{ch}_mask.png"
            mask_path = os.path.join(self.mask_dir, mask_file_name)
            if os.path.exists(mask_path):
                channel_mask = Image.open(mask_path).convert("L")
                mask.append(channel_mask)
            else:
                # If mask doesn't exist, append a zero mask of the same size as the image
                w, h = image.size
                mask.append(Image.new('L', (w, h)))

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            # We need to apply the transform to each mask and then stack them
            mask_tensors = [self.mask_transform(m) for m in mask]
            # Then we stack them
            mask = torch.cat(mask_tensors, dim=0)
        else:
            # If no transform, we need to convert to tensor manually
            mask = [transforms.ToTensor()(m) for m in mask]
            mask = torch.cat(mask, dim=0)

        return image, mask, img_name

def visualize_data(dataset, num_samples=5):
    for i in range(num_samples):
        image, mask, img_name = dataset[i]
        
        # Squeeze the dimensions for plotting
        image = transforms.ToTensor()(image)
        image = image.squeeze()
        mask = mask.squeeze(0)

        # Create plot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Plot input image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f"Input Image: {img_name}")
        axes[0].axis('off')

        # Get class names from the dataset
        class_names = dataset.class_names

        # Plot true masks
        for j in range(3):
            axes[j+1].imshow(mask[j], cmap='gray')
            axes[j+1].set_title(f"True Mask: {class_names[j]}")
            axes[j+1].axis('off')

        plt.tight_layout()
        plt.savefig(f"data_visualization_{i}.png")
        print(f"Saved data visualization to data_visualization_{i}.png")

if __name__ == '__main__':
    # Create dataset
    image_dir = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/processed/images"
    mask_dir = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/Segmentations/channels_cellpose"
    
    full_dataset = SpheroidSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)

    visualize_data(full_dataset)

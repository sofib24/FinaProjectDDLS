
import os
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import cv2
import matplotlib.image as mpimg

class SpheroidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        
        mask_paths = {
            'endothelial': os.path.join(self.mask_dir, f"{base_name}_ch002_mask.png"),
            'tumor': os.path.join(self.mask_dir, f"{base_name}_ch003_mask.png"),
            'fibroblasts': os.path.join(self.mask_dir, f"{base_name}_ch004_mask.png")
        }

        image = tiff.imread(img_path).astype(np.float32)
        image = cv2.resize(image, (256, 256))
        image = np.stack([image, image, image], axis=-1) # Convert to 3-channel
        
        masks = []
        for cell_type in ['endothelial', 'tumor', 'fibroblasts']:
            mask = mpimg.imread(mask_paths[cell_type])
            mask = cv2.resize(mask, (256, 256))
            masks.append(mask)
        
        mask = np.stack(masks, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

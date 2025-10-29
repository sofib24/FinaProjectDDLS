import os
import tifffile
import numpy as np
from skimage.color import rgb2gray
from skimage import io

# Create directories for processed data
os.makedirs("/content/drive/MyDrive/DDLS-Course/FinalProject/data/processed/images", exist_ok=True)
os.makedirs("/content/drive/MyDrive/DDLS-Course/FinalProject/data/processed/masks", exist_ok=True)

def process_images(file_paths):
    for file_path in file_paths:
        # Get the base name for the image
        base_name = os.path.basename(file_path).replace('_merge_all.tif', '')

        # Convert to grayscale and save
        image = tifffile.imread(file_path)
        grayscale_image = rgb2gray(image)
        grayscale_image_path = f"/content/drive/MyDrive/DDLS-Course/FinalProject/data/processed/images/{base_name}.tif"
        io.imsave(grayscale_image_path, grayscale_image)

        # Create and save the mask
        mask = np.zeros(grayscale_image.shape + (3,), dtype=np.uint8)
        for i, ch in enumerate(['ch001', 'ch002', 'ch003']):
            mask_path = file_path.replace('_merge_all.tif', f'_{ch}.tif')
            if os.path.exists(mask_path):
                mask_channel = rgb2gray(tifffile.imread(mask_path))
                mask[..., i] = mask_channel
        
        mask_save_path = f"/content/drive/MyDrive/DDLS-Course/FinalProject/data/processed/masks/{base_name}.tif"
        io.imsave(mask_save_path, mask)

# Get the file paths from the glob results
a_files = [
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_24h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_24h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_24h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_24h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_24h_Image005_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_48h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_48h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_48h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_48h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_48h_Image005_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_96h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_96h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_96h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_96h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/A_96h_Image005_merge_all.tif"
]

b_files = [
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_24h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_24h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_24h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_24h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_24h_Image005_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_48h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_48h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_48h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_48h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_48h_Image005_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_96h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_96h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_96h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_96h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/B_96h_Image005_merge_all.tif"
]

c_files = [
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_24h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_24h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_24h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_24h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_24h_Image005_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_48h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_48h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_48h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_48h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_48h_Image005_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_96h_Image001_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_96h_Image002_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_96h_Image003_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_96h_Image004_merge_all.tif",
    "/content/drive/MyDrive/DDLS-Course/FinalProject/data/MIP/C_96h_Image005_merge_all.tif"
]

process_images(a_files)
process_images(b_files)
process_images(c_files)

print("Data preparation complete.")
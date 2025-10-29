import os
import numpy as np
from PIL import Image

def convert_masks_to_8bit(source_directory, destination_directory):
    os.makedirs(destination_directory, exist_ok=True) # Ensure destination directory exists

    for filename in os.listdir(source_directory):
        if filename.endswith("_mask.png"):
            source_file_path = os.path.join(source_directory, filename)
            destination_file_path = os.path.join(destination_directory, filename)
            try:
                mask_data = Image.open(source_file_path)
                mask_array = np.array(mask_data)

                # Convert to 8-bit
                if mask_array.max() > 0:
                    mask_array_8bit = (mask_array / mask_array.max() * 255).astype(np.uint8)
                else:
                    mask_array_8bit = mask_array.astype(np.uint8) # All zeros

                # Save to the new destination directory
                Image.fromarray(mask_array_8bit).save(destination_file_path)
                print(f"Converted and saved {filename} to {destination_directory}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    source_dir = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/Segmentations/channels_cellpose"
    destination_dir = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/Segmentations/8bit"
    convert_masks_to_8bit(source_dir, destination_dir)
    print("Finished converting masks to 8-bit and saving to new folder.")
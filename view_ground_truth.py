import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from PIL import Image # Import Pillow for image manipulation

def view_ground_truth_masks(image_id):
    base_path = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/Segmentations/channels_cellpose"
    channel_suffixes = ['ch002', 'ch003', 'ch004']
    titles = ['Ground Truth (Endothelial)', 'Ground Truth (Tumor)', 'Ground Truth (Fibroblasts)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Ground Truth Masks for {image_id}', fontsize=16)

    for i, ch_suffix in enumerate(channel_suffixes):
        mask_name = f'{image_id}_{ch_suffix}_mask.png'
        mask_path = os.path.join(base_path, mask_name)

        if os.path.exists(mask_path):
            mask_img = mpimg.imread(mask_path)
            axes[i].imshow(mask_img, cmap='viridis')
            axes[i].set_title(titles[i])
            axes[i].axis('off')
        else:
            axes[i].set_title(f'{titles[i]} (Not Found)')
            axes[i].axis('off')
            print(f"Warning: Mask not found at {mask_path}")

    plt.tight_layout()
    # Save the plot to a file in the project's root directory
    output_path = "/content/drive/MyDrive/DDLS-Course/FinalProject/ground_truth_masks.png"
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free up memory
    print(f"Ground truth masks saved to: {output_path}")

    # --- New section to save one mask as 8-bit PNG to data/Segmentations/8bit ---
    # Let's pick the first mask (Endothelial) for demonstration
    example_mask_name = f'{image_id}_ch002_mask.png'
    example_mask_path = os.path.join(base_path, example_mask_name)

    if os.path.exists(example_mask_path):
        mask_data = mpimg.imread(example_mask_path)

        # Convert to 8-bit if not already
        if mask_data.dtype != np.uint8:
            # Scale to 0-255 and convert to uint8
            mask_data_8bit = (mask_data / mask_data.max() * 255).astype(np.uint8)
        else:
            mask_data_8bit = mask_data

        # Save the 8-bit mask to the new folder
        output_8bit_mask_path = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/Segmentations/8bit/ground_truth_ch002_8bit.png"
        Image.fromarray(mask_data_8bit).save(output_8bit_mask_path)
        print(f"8-bit ground truth ch002 mask saved to: {output_8bit_mask_path}")
    else:
        print(f"Warning: Example mask for 8-bit conversion not found at {example_mask_path}")
    # --- End of new section ---


if __name__ == '__main__':
    example_image_id = "A_24h_Image001"
    view_ground_truth_masks(example_image_id)
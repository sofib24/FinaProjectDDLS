import os
import io
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torchvision.models import resnet34

# Assuming UNetWithResnet34Encoder is defined in a separate file or copied here
# For now, I'll copy the class definition directly into main.py for simplicity.
# In a real project, this would be imported from a module.

class UNetWithResnet34Encoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.base_model = resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # Encoder
        self.encoder0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x/2, y/2)
        self.encoder1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x/4, y/4)
        self.encoder2 = self.base_layers[5]  # size=(N, 128, x/8, y/8)
        self.encoder3 = self.base_layers[6]  # size=(N, 256, x/16, y/16)
        self.encoder4 = self.base_layers[7]  # size=(N, 512, x/32, y/32)

        # Decoder
        self.decoder4 = self._decoder_block(512 + 256, 256)
        self.decoder3 = self._decoder_block(256 + 128, 128)
        self.decoder2 = self._decoder_block(128 + 64, 64)
        self.decoder1 = self._decoder_block(64 + 64, 64)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final convolution
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Decoder
        dec4 = self.upsample(enc4)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upsample(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upsample(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc0], dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)

        return nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)


app = FastAPI()

# Configuration
MODEL_PATH = "/content/drive/MyDrive/DDLS-Course/FinalProject/step3_unet_model/unet_spheroid.pth"
IMG_HEIGHT = 256
IMG_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROUND_TRUTH_MASK_DIR = "/content/drive/MyDrive/DDLS-Course/FinalProject/data/Segmentations/8bit"

# Mount static files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Configure Jinja2Templates
templates = Jinja2Templates(directory="webapp/templates")

# Load the model
model = UNetWithResnet34Encoder(n_classes=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def dice_coeff(pred, target):
    smooth = 1.
    # Ensure pred and target are flattened and on the same device
    m1 = pred.view(-1)
    m2 = target.view(-1)
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L") # Convert to grayscale
    original_image_np = np.array(image)

    # Preprocess image for model
    image_resized_for_model = cv2.resize(original_image_np, (IMG_WIDTH, IMG_HEIGHT))
    image_float = image_resized_for_model.astype(np.float32) / 255.0
    image_3_channel = np.stack([image_float, image_float, image_float], axis=0) # C, H, W
    image_tensor = torch.from_numpy(image_3_channel).unsqueeze(0).to(DEVICE) # Add batch dim

    # Inference
    with torch.no_grad():
        prediction = model(image_tensor)
    
    pred_mask_tensor = (torch.sigmoid(prediction) > 0.05).float()
    pred_mask_np = pred_mask_tensor.cpu().numpy().squeeze(0) # Remove batch dim, C, H, W

    # Load ground truth masks
    filename_base = os.path.splitext(file.filename)[0]
    gt_masks = []
    dice_scores = {}
    cell_types = ["endothelial", "tumor", "fibroblasts"]
    gt_mask_paths = [
        os.path.join(GROUND_TRUTH_MASK_DIR, f"{filename_base}_ch002_mask.png"),
        os.path.join(GROUND_TRUTH_MASK_DIR, f"{filename_base}_ch003_mask.png"),
        os.path.join(GROUND_TRUTH_MASK_DIR, f"{filename_base}_ch004_mask.png"),
    ]

    for i, gt_path in enumerate(gt_mask_paths):
        if os.path.exists(gt_path):
            gt_mask_img = Image.open(gt_path).convert("L")
            gt_mask_np = np.array(gt_mask_img) / 255.0 # Normalize to 0-1
            gt_mask_resized = cv2.resize(gt_mask_np, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            gt_masks.append(gt_mask_resized)
            
            # Calculate Dice score
            pred_flat = pred_mask_tensor[0, i].cpu()
            gt_flat = torch.from_numpy(gt_mask_resized).float()
            dice = dice_coeff(pred_flat, gt_flat)
            dice_scores[cell_types[i]] = f"{dice:.4f}"
        else:
            gt_masks.append(np.zeros((IMG_HEIGHT, IMG_WIDTH)))
            dice_scores[cell_types[i]] = "N/A"

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Original Image
    axes[0, 0].imshow(original_image_np, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Predicted Masks
    titles_pred = ['Pred Endothelial', 'Pred Tumor', 'Pred Fibroblasts']
    for i in range(3):
        axes[0, i+1].imshow(pred_mask_np[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i+1].set_title(f'{titles_pred[i]} (Dice: {dice_scores[cell_types[i]]})')
        axes[0, i+1].axis('off')

    # Ground Truth Masks
    axes[1, 0].axis('off') # Empty for alignment
    titles_gt = ['GT Endothelial', 'GT Tumor', 'GT Fibroblasts']
    for i in range(3):
        axes[1, i+1].imshow(gt_masks[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i+1].set_title(titles_gt[i])
        axes[1, i+1].axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Load loss plot image and encode to base64
    loss_plot_path = "/content/drive/MyDrive/DDLS-Course/FinalProject/step3_unet_model/loss_plot.png"
    encoded_loss_plot = None
    if os.path.exists(loss_plot_path):
        with open(loss_plot_path, "rb") as f:
            encoded_loss_plot = base64.b64encode(f.read()).decode('utf-8')

    return JSONResponse(content={
        "filename": file.filename,
        "image_base64": image_base64,
        "dice_scores": dice_scores,
        "loss_plot_base64": encoded_loss_plot,
        "message": "Processing complete. Predicted masks generated."
    })


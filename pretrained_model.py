import torch
import torch.nn as nn
from torchvision.models import resnet34

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
        self.decoder4 = self.decoder_block(512 + 256, 256)
        self.decoder3 = self.decoder_block(256 + 128, 128)
        self.decoder2 = self.decoder_block(128 + 64, 64)
        self.decoder1 = self.decoder_block(64 + 64, 64)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final convolution
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def decoder_block(self, in_channels, out_channels):
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
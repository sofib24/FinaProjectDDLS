import torch

state_dict = torch.load('../step3_unet_model/unet_spheroid.pth', map_location=torch.device('cpu'))
for key in state_dict.keys():
    print(key)

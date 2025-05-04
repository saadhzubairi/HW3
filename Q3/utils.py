import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import config

# Initialize weights for Conv2d, ConvTranspose2d, and Linear layers
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)

# Save generated samples as a grid image
def save_samples(gen, epoch, fixed_noise):
    gen.eval()  # Set generator to evaluation mode
    # Disable gradient computation
    with torch.no_grad():  
        # Normalize to [0,1]
        imgs = gen(fixed_noise).add(1).div(2)  
        # Create grid of images
        grid = make_grid(imgs, nrow=8)  
        # Ensure output directory exists
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)  
        # Save grid image
        save_image(grid, f"{config.OUTPUT_DIR}/epoch_{epoch}.png")  
    # Restore generator to training mode
    gen.train()  

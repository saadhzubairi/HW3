import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import config

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)

def save_samples(gen, epoch, fixed_noise):
    gen.eval()
    with torch.no_grad():
        imgs = gen(fixed_noise).add(1).div(2)  # from [-1,1] → [0,1]
        grid = make_grid(imgs, nrow=8)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        save_image(grid, f"{config.OUTPUT_DIR}/epoch_{epoch}.png")
    gen.train()

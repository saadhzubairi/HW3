import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from models import Generator, Discriminator
from utils import weights_init, save_samples

def train():
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = get_dataloader()
    gen  = Generator().to(device)
    disc = Discriminator().to(device)
    gen.apply(weights_init)
    disc.apply(weights_init)

    opt_g = optim.Adam(gen.parameters(), lr=config.LR, betas=config.BETAS)
    opt_d = optim.Adam(disc.parameters(), lr=config.LR, betas=config.BETAS)
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, config.LATENT_DIM, device=device)
    writer = SummaryWriter(config.LOG_DIR)

    step = 0
    for ep in range(1, config.EPOCHS+1):
        loop = tqdm(dl, desc=f"Epoch {ep}/{config.EPOCHS}")
        for real, _ in loop:
            real = real.to(device)
            bs = real.size(0)
            noise = torch.randn(bs, config.LATENT_DIM, device=device)
            fake  = gen(noise)

            # Discriminator step
            opt_d.zero_grad()
            d_real = disc(real)
            d_fake = disc(fake.detach())
            loss_d = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            loss_d.backward()
            opt_d.step()

            # Generator step
            opt_g.zero_grad()
            d_fake2 = disc(fake)
            loss_g = criterion(d_fake2, torch.ones_like(d_fake2))
            loss_g.backward()
            opt_g.step()

            writer.add_scalar("Loss/Discriminator", loss_d.item(), step)
            writer.add_scalar("Loss/Generator",     loss_g.item(), step)
            step += 1
            loop.set_postfix(d_loss=loss_d.item(), g_loss=loss_g.item())

        if ep in config.SAVE_EPOCHS:
            save_samples(gen, ep, fixed_noise)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.save(gen.state_dict(), os.path.join(config.OUTPUT_DIR, "generator.pth"))
    torch.save(disc.state_dict(), os.path.join(config.OUTPUT_DIR, "discriminator.pth"))
    writer.close()

if __name__ == "__main__":
    train()

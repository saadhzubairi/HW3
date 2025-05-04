import torch
from models import Generator
from utils import save_samples
import config

def generate(model_path=None, n=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    mp = model_path or f"{config.OUTPUT_DIR}/generator.pth"
    G.load_state_dict(torch.load(mp, map_location=device))
    noise = torch.randn(n, config.LATENT_DIM, device=device)
    save_samples(G, "final", noise)

if __name__ == "__main__":
    generate()

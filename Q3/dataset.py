import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

def get_dataloader():
    tf = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.FashionMNIST(
        root=config.DATA_DIR, train=True, download=True, transform=tf
    )
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

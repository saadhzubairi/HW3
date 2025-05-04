import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

# Function to create and return a DataLoader for the FashionMNIST dataset
def get_dataloader():
    # Define a series of transformations to preprocess the images
    tf = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),  
        transforms.ToTensor(),              
        transforms.Normalize((0.5,), (0.5,)),  
    ])
    
    # Load the FashionMNIST dataset with the specified transformations
    ds = datasets.FashionMNIST(
        root=config.DATA_DIR,  
        train=True,           
        download=True,        
        transform=tf          
    )
    
    # Create and return a DataLoader for the dataset
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,  
        shuffle=True,                  
        num_workers=4,                 
        pin_memory=True,               
    )

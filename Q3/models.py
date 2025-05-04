import torch.nn as nn
import config

# Discriminator: A CNN-based model to classify real vs. fake images
class Discriminator(nn.Module):
    def __init__(self, channels=config.CHANNELS, feat=64, slope=0.3, drop=0.3):
        super().__init__()
        # Convolutional layers to downsample the input image
        self.conv = nn.Sequential(
            nn.Conv2d(channels, feat, 5, 2, 2, bias=False),  # 1x28x28 -> 64x14x14
            nn.LeakyReLU(slope, inplace=True),              # Leaky ReLU activation
            nn.Dropout(drop),                               # Dropout for regularization
            nn.Conv2d(feat, feat*2, 5, 2, 2, bias=False),   # 64x14x14 -> 128x7x7
            nn.LeakyReLU(slope, inplace=True),              # Leaky ReLU activation
            nn.Dropout(drop),                               # Dropout for regularization
        )
        # Fully connected layer to map the flattened feature map to a scalar
        ds = config.IMG_SIZE // 4  # Downsampled size (IMG_SIZE / 2^2 due to 2 conv layers)
        self.fc = nn.Linear(feat*2*ds*ds, 1, bias=False)  # 128x7x7 -> scalar

    def forward(self, x):
        x = self.conv(x)  # Apply convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the feature map
        return self.fc(x).view(-1)  # Output a scalar for each input in the batch

# Generator: A CNN-based model to generate images from random noise
class Generator(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM, channels=config.CHANNELS, feat=64, slope=0.3):
        super().__init__()
        ds = config.IMG_SIZE // 4  # Downsampled size (7x7 for 28x28 images)
        # Fully connected layer to project latent vector to a feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, feat*4*ds*ds, bias=False),  # 100 -> 256x7x7
            nn.BatchNorm1d(feat*4*ds*ds),                    # Batch normalization
            nn.LeakyReLU(slope, inplace=True),               # Leaky ReLU activation
        )
        # Transposed convolutional layers to upsample the feature map
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (feat*4, ds, ds)),               # Reshape to 256x7x7
            nn.ConvTranspose2d(feat*4, feat*2, 5, 2, 2, output_padding=1, bias=False),  # 256x7x7 -> 128x7x7
            nn.BatchNorm2d(feat*2),                          # Batch normalization
            nn.LeakyReLU(slope, inplace=True),               # Leaky ReLU activation
            nn.ConvTranspose2d(feat*2, feat, 5, 2, 2, output_padding=1, bias=False),    # 128x7x7 -> 64x14x14
            nn.BatchNorm2d(feat),                            # Batch normalization
            nn.LeakyReLU(slope, inplace=True),               # Leaky ReLU activation
            nn.ConvTranspose2d(feat, channels, 5, 1, 2, bias=False),  # 64x14x14 -> 1x28x28
            nn.Tanh(),                                       # Tanh activation for output
        )

    def forward(self, z):
        x = self.fc(z)  # Project latent vector to feature map
        return self.deconv(x)  # Upsample to generate an image

import torch.nn as nn
import config

class Discriminator(nn.Module):
    def __init__(self, channels=config.CHANNELS, feat=64, slope=0.3, drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, feat, 5, 2, 2, bias=False),
            nn.LeakyReLU(slope, inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(feat, feat*2, 5, 2, 2, bias=False),
            nn.LeakyReLU(slope, inplace=True),
            nn.Dropout(drop),
        )
        ds = config.IMG_SIZE // 4
        self.fc = nn.Linear(feat*2*ds*ds, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).view(-1)

class Generator(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM, channels=config.CHANNELS, feat=64, slope=0.3):
        super().__init__()
        ds = config.IMG_SIZE // 4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, feat*4*ds*ds, bias=False),
            nn.BatchNorm1d(feat*4*ds*ds),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (feat*4, ds, ds)),
            nn.ConvTranspose2d(feat*4, feat*2, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(feat*2),
            nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(feat*2, feat, 5, 2, 2, output_padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(feat, channels, 5, 1, 2, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        return self.deconv(x)

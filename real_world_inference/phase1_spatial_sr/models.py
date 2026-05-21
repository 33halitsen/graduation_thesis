import torch.nn as nn

class ESPCN_Baseline(nn.Module):
    def __init__(self, upscale_factor=3):
        super(ESPCN_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

class FastResidualBlock(nn.Module):
    def __init__(self, channels):
        super(FastResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class FastResESPCN(nn.Module):
    def __init__(self, upscale_factor=3):
        super(FastResESPCN, self).__init__()
        self.conv_in = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.res_block1 = FastResidualBlock(64)
        self.res_block2 = FastResidualBlock(64)
        self.conv_out = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_ps = nn.Conv2d(32, 1 * (upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv_in(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.relu(self.conv_out(x))
        x = self.conv_ps(x)
        x = self.pixel_shuffle(x)
        return x

class ESPCN_Hybrid(nn.Module):
    def __init__(self, upscale_factor=3):
        super(ESPCN_Hybrid, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

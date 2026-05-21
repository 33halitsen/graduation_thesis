import torch.nn as nn


class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (ESPCN)
    for Real-Time Super-Resolution.
    """

    def __init__(self, upscale_factor=3):
        super(ESPCN, self).__init__()
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

import torch
import torch.nn as nn

class MicroVFINetPro(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.ReLU(True))
        self.drop = nn.Dropout2d(p=0.2)
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True))
        self.final = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

    def get_coord_grids(self, x):
        b, c, h, w = x.size()
        y_grid = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x_grid, y_grid], dim=1)

    def forward(self, im1, im3):
        x = torch.cat([im1, im3], dim=1)
        e1 = self.enc1(torch.cat([x, self.get_coord_grids(x)], dim=1))
        e2 = self.drop(self.enc2(e1))
        e3 = self.enc3(e2)
        d2 = self.dec2(e3)
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        return self.final(torch.cat([d1, e1], dim=1))

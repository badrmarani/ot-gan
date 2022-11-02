import torch
from torch import nn

class CReLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super(CReLU, self).__init__()
        self.dim = dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = torch.cat((x, -x), dim=self.dim)
        return nn.functional.relu(out)

    def extra_repr(self) -> str:
        return 'dim={}'.format(self.dim)

class Critic(nn.Module):
    def __init__(self,
        n_base_channels: int = 3
        ) -> None:
        super(Critic, self).__init__()
        self.net = []
        self.net.append(self._block(n_base_channels, 128, stride=1))
        self.net.append(self._block(256, 256, stride=2))
        self.net.append(self._block(512, 512, stride=2))
        self.net.append(self._block(1024, 1024, stride=2))
        self.net = nn.Sequential(*self.net)

    def _block(self, in_size: int, out_size: int, stride: int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=5, stride=stride, padding=2),
            CReLU(dim=1),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.net(x)
        out = torch.flatten(out, start_dim=1)
        out = nn.functional.normalize(out, p=2, dim=1)
        return out

class Generator(nn.Module):
    def __init__(self,
        n_base_channels: int = 3
        ) -> None:
        super(Generator, self).__init__()
        self.l1 = nn.Linear(100, 1024*4*4*2)
        self.conv = nn.Conv2d(128, n_base_channels, kernel_size=(5,5), stride=1, padding=2)

        self.net = []
        self.net.append(self._block(1024))
        self.net.append(self._block(1024//2))
        self.net.append(self._block(1024//4))
        self.net = nn.Sequential(*self.net)
        

    def _block(self, out_size: int):
        return nn.Sequential(
            nn.Upsample(scale_factor=(2,2), mode="nearest"),
            nn.Conv2d(out_size, out_size, kernel_size=(5,5), stride=1, padding=2),
            nn.GLU(dim=1),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = nn.functional.glu(self.l1(x), dim=1).view(-1, 1024, 4, 4)
        out = self.net(out)
        out = torch.tanh(self.conv(out))
        return out
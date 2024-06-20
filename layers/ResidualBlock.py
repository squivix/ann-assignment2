from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )
        self.residual = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block1.forward(x) + self.residual.forward(x)

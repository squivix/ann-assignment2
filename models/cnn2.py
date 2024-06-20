from torch import nn

from layers.ResidualBlock import ResidualBlock


class CnnModel2(nn.Module):
    def __init__(self, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            ResidualBlock(1, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(50, output_dim),
        )

    def forward(self, x):
        return self.model.forward(x[:, None, :, :])

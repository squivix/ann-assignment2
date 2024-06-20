from torch import nn

from layers.ResidualBlock import ResidualBlock


class CnnModel2(nn.Module):
    def __init__(self, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            # nn.Conv2d(1, 16, kernel_size=5, stride=3, padding=1),
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2),
            ResidualBlock(1, 16),
            ResidualBlock(16, 16),
            # ResidualBlock(32, 32),
            nn.Conv2d(16, 16, kernel_size=11, stride=3, padding="valid"),
            nn.AvgPool2d(kernel_size=3, stride=1),

            nn.Flatten(start_dim=1),
            nn.Linear(256, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(20, output_dim),
        )

    def forward(self, x):
        return self.model.forward(x[:, None, :, :])

from torch import nn


class CnnModel1(nn.Module):
    def __init__(self, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=8, stride=1, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1),
            nn.Linear(144, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, output_dim),
        )

    def forward(self, x):
        return self.model.forward(x[:, None, :, :])

from torch import nn


class CnnModel1(nn.Module):
    def __init__(self, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1),
            nn.Linear(400, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(50, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(20, output_dim),
        )

    def forward(self, x):
        return self.model.forward(x[:, None, :, :])

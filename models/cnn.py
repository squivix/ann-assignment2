from torch import nn


class CnnModel1(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=15, kernel_size=3, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=15, out_channels=15, kernel_size=3, stride=2, padding="valid"),
            nn.Flatten(start_dim=1),
            nn.Linear(4 * 15, output_dim),
        )

    def forward(self, x):
        return self.model.forward(x)

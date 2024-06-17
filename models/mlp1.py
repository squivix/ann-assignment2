from torch import nn


class MlpModel1(nn.Module):
    def __init__(self, input_dim_w, input_dim_h, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim_w * input_dim_h, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, output_dim),
        )

    def forward(self, x):
        return self.model.forward(x)

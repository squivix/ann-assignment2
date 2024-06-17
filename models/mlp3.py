from torch import nn


class MlpModel3(nn.Module):
    def __init__(self, input_dim_w, input_dim_h, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(

            nn.Flatten(),
            nn.Linear(input_dim_w * input_dim_h, 200),
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
            nn.Linear(50, 25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(25, output_dim),
        )

    def forward(self, x):
        return self.model.forward(x)

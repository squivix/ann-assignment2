from torch import nn


class MlpModel2(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(

        )

    def forward(self, x):
        return self.model.forward(x)

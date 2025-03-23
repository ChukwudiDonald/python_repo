import torch.nn as nn

class CypherNet(nn.Module):
    def __init__(self, **kwargs):
        super(CypherNet, self).__init__()
        n_chars = kwargs.get('n_chars', 26)
        self.fc = nn.Linear(n_chars, 128)
        self.fo = nn.Linear(128, n_chars)

        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.activation(self.fc(x))
        x = self.activation(self.fo(x))

        return x
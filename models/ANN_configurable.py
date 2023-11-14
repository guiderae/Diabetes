import torch
import torch.nn as nn
import torch.nn.functional as F


class ANNModelConfigurable(nn.Module):
    def __init__(self, hidden_layers=None, epochs=None, learn_rate=None):
        super().__init__()
        self.input_features = 8
        self.out_features = 2
        self.f_connected1 = None
        self.f_connected2 = None
        self.f_connected3 = None
        self.f_connected4 = None
        self.out = None
        if hidden_layers is not None:
            self.f_connected1 = nn.Linear(self.input_features, hidden_layers[0])
            self.f_connected2 = nn.Linear(hidden_layers[0], hidden_layers[1])
            self.f_connected3 = None
            self.f_connected4 = None

            if hidden_layers[2] == 0:
                self.out = nn.Linear(hidden_layers[1], self.out_features)
            else:
                self.f_connected3 = nn.Linear(hidden_layers[1], hidden_layers[2])
                if hidden_layers[3] == 0:
                    self.out = nn.Linear(hidden_layers[2], self.out_features)
                else:
                    self.f_connected4 = nn.Linear(hidden_layers[2], hidden_layers[3])
                    self.out = nn.Linear(hidden_layers[3], self.out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        if self.f_connected3 is not None:
            x = F.relu(self.f_connected3(x))
            if self.f_connected4 is not None:
                x = F.relu(self.f_connected4(x))
        x = self.out(x)
        return x

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    @staticmethod
    def load_model(file_path):
        model = torch.load(file_path)
        return model

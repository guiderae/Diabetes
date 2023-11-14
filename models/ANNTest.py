import torch
import torch.nn as nn
import torch.nn.functional as F


class ANNModel(nn.Module):
    def __init__(self, input_features=8, hidden1=24, hidden2=12, hidden3=0, hidden4=0, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.f_connected3 = None
        self.f_connected4 = None

        if hidden3 == 0 :
            self.out = nn.Linear(hidden2, out_features)
        else:
            self.f_connected3 = nn.Linear(hidden2, hidden3)
            if hidden4 == 0:
                self.out = nn.Linear(hidden3, out_features)
            else:
                self.f_connected4 = nn.Linear(hidden3, hidden4)
                self.out = nn.Linear(hidden4, out_features)

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
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()
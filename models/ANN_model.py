import torch
import torch.nn as nn
import torch.nn.functional as F


# The Artificial Neural Network Model with 8 input nodes, and two hidden layers with 24 and 12 nodes respectively,
# then 2 output nodes.
class ANNModel(nn.Module):
    def __init__(self, input_features, hidden1, hidden2, out_features):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    @staticmethod
    def load_model(file_path):
        model = torch.load(file_path)
        #state_dict = nn.ModuleDict.state_dict(file_path)
        #model = nn.Module.load_state_dict(torch.load(file_path))
        #model = nn.Module.load_state_dict(nn.Module.state_dict(file_path))

        #model = keras.models.load_model(file_path)
        return model

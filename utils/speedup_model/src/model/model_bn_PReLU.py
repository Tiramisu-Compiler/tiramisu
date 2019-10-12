import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_BN_PReLU(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[10, 10, 10], drops=[0.4, 0.4, 0.4]):
        super().__init__()
        hidden_sizes = [input_size] + hidden_sizes

        self.hidden_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropouts= nn.ModuleList()

        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=False))
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            nn.init.xavier_uniform_(self.hidden_layers[i].weight)
            self.dropouts.append(nn.Dropout(drops[i]))


        self.predict = nn.Linear(hidden_sizes[-1], output_size)
        nn.init.xavier_uniform_(self.predict.weight)
        self.PReLU=nn.PReLU()

        

    def forward(self, x):

        for i in range(len(self.hidden_layers)):
            x = self.dropouts[i](self.PReLU(self.batch_norm_layers[i](self.hidden_layers[i](x))))

        x = self.predict(x)
        return F.relu(x)
    
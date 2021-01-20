import torch
from torch import nn
import torch.nn.functional as F
from operator import *

class Model_hier_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[10, 10, 10], drops=[0.4, 0.4, 0.4]):
        super().__init__()
        rnn_hidden_state_size = hidden_sizes[-1]
        hidden_sizes2 = [rnn_hidden_state_size] + hidden_sizes[-2:]
        concat_hidden_sizes = [rnn_hidden_state_size*2] + [200, 180]
        hidden_sizes = [input_size] + hidden_sizes
        self.hidden_layers = nn.ModuleList()
        self.dropouts= nn.ModuleList()
        self.hidden_layers2 = nn.ModuleList()
        self.dropouts2= nn.ModuleList()
        self.concat_hidden_layers = nn.ModuleList()
        self.concat_dropouts= nn.ModuleList()
        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=False))
            nn.init.xavier_uniform_(self.hidden_layers[i].weight)
            self.dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(hidden_sizes2)-1):
            self.hidden_layers2.append(nn.Linear(hidden_sizes2[i], hidden_sizes2[i+1], bias=False))
            nn.init.xavier_uniform_(self.hidden_layers2[i].weight)
            self.dropouts2.append(nn.Dropout(drops[i]))
        for i in range(len(concat_hidden_sizes)-1):
            self.concat_hidden_layers.append(nn.Linear(concat_hidden_sizes[i], concat_hidden_sizes[i+1], bias=False))
            nn.init.xavier_uniform_(self.concat_hidden_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        self.predict = nn.Linear(hidden_sizes2[-1], output_size)
        nn.init.xavier_uniform_(self.predict.weight)
        self.ELU=nn.ELU()
        self.no_comps_tensor = nn.Parameter(torch.zeros(1, rnn_hidden_state_size))
        self.no_nodes_tensor = nn.Parameter(torch.zeros(1, rnn_hidden_state_size))
        self.comps_lstm = nn.LSTM(hidden_sizes[-1], rnn_hidden_state_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(hidden_sizes[-1], rnn_hidden_state_size, batch_first=True)
        
    def get_hidden_state(self, node, tensor):
        nodes_list = []
        for n in node['child_list']:
            nodes_list.append(self.get_hidden_state(n, tensor))
        if (nodes_list != []):
            nodes_tensor = torch.cat(nodes_list, 1) 
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1,0,2)
        else:       
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(tensor.shape[0], -1, -1)
        if (node['has_comps']==1):
            comps_tensor = torch.index_select(tensor, 1, node['computations_indices'])
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(comps_tensor) 
            comps_h_n = comps_h_n.permute(1,0,2)
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(tensor.shape[0], -1, -1)
        x = torch.cat((nodes_h_n, comps_h_n),2)
        for i in range(len(self.concat_hidden_layers)):
            x = self.concat_hidden_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x  

    def forward(self, tree_tensor):
        tree, tensor = tree_tensor
        x = tensor
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = self.dropouts[i](self.ELU(x))    
        x = self.get_hidden_state(tree, x)
        for i in range(len(self.hidden_layers2)):
            x=self.hidden_layers2[i](x)
            x = self.dropouts2[i](self.ELU(x))
        out = self.predict(x)
            
        return F.relu(out[:,0,0])

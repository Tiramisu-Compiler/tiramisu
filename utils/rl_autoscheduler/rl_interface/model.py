import math
import torch
import json
import traceback
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from tiramisu_programs.surrogate_model_utils.json_to_tensor import get_tree_footprint

train_device_name = 'cpu'  # choose training/storing device, either 'cuda:X' or 'cpu'
store_device_name = 'cpu'


store_device = torch.device(store_device_name)
train_device = torch.device(train_device_name)

BIG_NUMBER = 1e10

torch, nn = try_import_torch()


class TiramisuModelMult(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,**kwargs):

        print("in model init")

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name,**kwargs)

        nn.Module.__init__(self)
        
        shared_layer_sizes = model_config["custom_model_config"]["layer_sizes"]
        
        

        embedding_size= shared_layer_sizes[-1]

        num_outputs=action_space.n

        #Computation Embedding Layer
        prev_layer_size = obs_space["representation"].shape[1]
        comp_embd_layers=[]      
        cpt=0
        for size in shared_layer_sizes:
            comp_embd_layers.extend(
                [nn.Linear(
                    prev_layer_size,
                    size
                ),
                nn.Dropout(0.02),
                nn.ELU()
                ]
            )
            prev_layer_size = size
            cpt+=1
        
        self._comp_embd_layers = nn.Sequential(*comp_embd_layers)
       #Recursive Loop Embedding Layer
        self.comps_lstm = nn.LSTM(shared_layer_sizes[-1], embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(shared_layer_sizes[-1], embedding_size, batch_first=True)
        
        self.no_comps_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, embedding_size))
        )
        self.no_nodes_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, embedding_size))
        )

        prev_layer_size=embedding_size*2+26
        hidden_layer_sizes = shared_layer_sizes[-2:]
        rec_loop_embd_layers=[]

        for size in hidden_layer_sizes:
            rec_loop_embd_layers.extend(
                [nn.Linear(
                    prev_layer_size,
                    size
                ),
                nn.Dropout(0.02),
                nn.ELU()
                ]
            )
            prev_layer_size = size
        
        self._rec_loop_embd_layers = nn.Sequential(*rec_loop_embd_layers)
        #Prediction Layer
        predict_layers=[]
        
        for size in hidden_layer_sizes:
            predict_layers.extend(
                [nn.Linear(
                    prev_layer_size,
                    size
                ),
                nn.Dropout(0.02),
                nn.ELU()
                ]
            )
            prev_layer_size = size
        
        self._prediction_layers = nn.Sequential(*predict_layers)

        #Outputs
        #1 Policy
        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

        #2 Value
        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )


    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_len):
        obs = input_dict["obs_flat"]["representation"]
    
        #computation embedding layer
        comps_embeddings=self._comp_embd_layers(obs)
        
        #recursive loop embedding layer
        loops_tensor=input_dict["obs_flat"]["loops_representation"]
        child_list=input_dict["obs_flat"]["child_list"][:][:][0][0]
        has_comps=input_dict["obs_flat"]["has_comps"][0].tolist()
        try:
            prog_tree_tensor = np.array(input_dict["obs_flat"]["prog_tree"])
            prog_tree_string = "".join(list(prog_tree_tensor[0].view('U1'))).strip("_")
            prog_tree = json.loads(prog_tree_string)
            # tree_footprint = get_tree_footprint(prog_tree)
        except:
            pass
        # prog_tree_string =  # FIX THIS  --> Compare tree footprint
        computations_indices=input_dict["obs_flat"]["computations_indices"][:][:][0][0]


        try:
            loop_index=0
            prog_embedding=self.get_hidden_state(prog_tree,comps_embeddings,loops_tensor)
        except:
            print("Actor Critic",traceback.format_exc())
            prog_tree = {"child_list":[]}
            prog_embedding = torch.zeros((loops_tensor.shape[0],180))
        
        # prediction layer
        self._features=self._prediction_layers(prog_embedding.view(prog_embedding.shape[0],-1))
        logits = self._logits(self._features)
        logits=logits-BIG_NUMBER*(1-input_dict["obs_flat"]["action_mask"])


        self._value = self._value_branch(self._features)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._value.squeeze(1)

    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            nodes_list.append(self.get_hidden_state(n, comps_embeddings, loops_tensor))
        if nodes_list != []:
            nodes_tensor = torch.cat(nodes_list, 1)
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        else:
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(
                comps_embeddings.shape[0], -1, -1
            )
        if node["has_comps"]:
            selected_comps_tensor = torch.index_select(
                comps_embeddings, 1, torch.tensor(node["computations_indices"])
            )
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(selected_comps_tensor)
            comps_h_n = comps_h_n.permute(1, 0, 2)
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                comps_embeddings.shape[0], -1, -1
            )
        selected_loop_tensor = torch.index_select(
            loops_tensor, 1, torch.tensor(node["loop_index"])
        )
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor), 2)
        x = self._rec_loop_embd_layers(x)
        return x
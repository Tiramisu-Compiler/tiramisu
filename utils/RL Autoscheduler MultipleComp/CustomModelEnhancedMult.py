import math
import torch
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

train_device_name = 'cpu' # choose training/storing device, either 'cuda:X' or 'cpu'
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
        prev_layer_size = int(np.product(obs_space["representation"].shape))
        comp_embd_layers=[]      
        cpt=0
        for size in shared_layer_sizes:
            if cpt==3:
                size=900 # the output layer size = nb_comps * embedding size (which is = 5 * 180). Note: nb_comps and embedding size can be modified if needed.
            comp_embd_layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn="relu",
                )
            )
            prev_layer_size = size
            cpt+=1
        
        self._comp_embd_layers = nn.Sequential(*comp_embd_layers)

       #Recursive Loop Embedding Layer
        self.comps_lstm = nn.LSTM(shared_layer_sizes[-1], embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(shared_layer_sizes[-1], embedding_size, batch_first=True)

        self.no_comps_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        self.no_nodes_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))

        prev_layer_size=embedding_size*2+26
        hidden_layer_sizes = shared_layer_sizes[-2:]
        rec_loop_embd_layers=[]

        for size in hidden_layer_sizes:
            rec_loop_embd_layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn="relu",
                )
            )
            prev_layer_size = size
        
        self._rec_loop_embd_layers = nn.Sequential(*rec_loop_embd_layers)

        #Prediction Layer
        predict_layers=[]
        
        for size in hidden_layer_sizes:
            predict_layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn="relu",
                )
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
        print("in forward")
        print('OBS:',input_dict['obs_flat'])
        obs = input_dict["obs_flat"]["representation"][-1,:,:]
        #OBS needs to be flattened because it has a shape of (5,1052) initially, as specified in the observation space
        obs=torch.flatten(obs)
    
        #computation embedding layer
        comps_embeddings=self._comp_embd_layers(obs)
        #print("from comp embd layer: ",comps_embeddings.size(),comps_embeddings)
        
        #recursive loop embedding layer
        loops_tensor=input_dict["obs_flat"]["loops_representation"]
        child_list=input_dict["obs_flat"]["child_list"][:][:][0][0]
        has_comps=input_dict["obs_flat"]["has_comps"][0].tolist()
        computations_indices=input_dict["obs_flat"]["computations_indices"][:][:][0][0]
        loop_index=0
        prog_embedding=self.get_hidden_state(input_dict["obs_flat"],child_list,has_comps,computations_indices, comps_embeddings, loops_tensor,loop_index)
        #print("from recursive embd layer: ",prog_embedding.size())
        
        # prediction layer
        self._features=self._prediction_layers(prog_embedding)
        #print("from prediction embd layer: ",self._features.size())
        logits = self._logits(self._features)
        logits=logits-BIG_NUMBER*(1-input_dict["obs_flat"]["action_mask"])

        print("\nLa distribution de probabilté est: ", F.softmax(logits))

        self._value = self._value_branch(self._features)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        print(self._value)
        return self._value.squeeze(1)
    
    def get_hidden_state(self, input_dict, child_list, has_comps, computations_indices, comps_embeddings, loops_tensor,loop_index):
        nodes_list = []

        if len(child_list)!=0 and not all(element == 0 for element in child_list):
            for child in child_list:
                tmp=math.floor(child)
                childs=input_dict["child_list"][:][:][0][tmp]
                #print('children',childs)
                computations_indices=input_dict["computations_indices"][:][:][0][tmp]
                #computations_indices=torch.tensor(computations_indices).to(train_device)
                childs=childs[childs!=-1]
                if len(childs)!=0:
                    nodes_list.append(self.get_hidden_state(input_dict,childs, has_comps, computations_indices, comps_embeddings,loops_tensor,tmp))

            if (nodes_list != []):
                nodes_tensor = torch.cat(nodes_list, 1) 
                if nodes_tensor.shape[1]!=180:
                    nodes_tensor=nodes_tensor.reshape(int(nodes_tensor.shape[1]/180),180)
                lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
                #print('1 nodes_h_n',nodes_h_n.size())
            else: 
                nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
                nodes_h_n=nodes_h_n[0]
                #print('2 nodes_h_n',nodes_h_n.size())
                
            if has_comps[loop_index]==1:
                computations_indices=computations_indices[computations_indices!=-1].int()
                comps_embeddings=comps_embeddings.reshape(5,180)
                selected_comps_tensor = torch.index_select(comps_embeddings, 0, computations_indices)
                lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(selected_comps_tensor) 
                #print('1 comps_h_n',comps_h_n.size())
            else:
                comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
                comps_h_n=comps_h_n[0]
                #print('2 comps_h_n',comps_h_n.size())

            selected_loop_tensor = torch.index_select(loops_tensor[-1,:,:],0,torch.tensor(loop_index).to(train_device))
            #selected_loop_tensor=selected_loop_tensor.resize_(1,1,24)
            #print('selected_loop_tensor',selected_loop_tensor.size())
            x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor),1)

            x= self._rec_loop_embd_layers(x)

        elif all(element == 0 for element in child_list):
            x=torch.zeros([1,180])
    
        return x


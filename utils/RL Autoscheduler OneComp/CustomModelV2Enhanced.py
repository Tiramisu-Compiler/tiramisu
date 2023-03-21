from os import stat
import torch

from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.utils.annotations import override, PublicAPI
import gym

import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

train_device_name = 'cpu' # choose training/storing device, either 'cuda:X' or 'cpu'
store_device_name = 'cpu'

store_device = torch.device(store_device_name)
train_device = torch.device(train_device_name)

BIG_NUMBER = 1e10

torch, nn = try_import_torch()

class TiramisuModelV2(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,**kwargs):

        print("in model init")

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name,**kwargs)

        nn.Module.__init__(self)
        

        shared_layer_sizes = model_config["custom_model_config"]["layer_sizes"]

        layers=[]
        prev_layer_size = int(np.product(obs_space["representation"].shape))

        num_outputs=53

        for size in shared_layer_sizes:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn="relu",
                )
            )
            prev_layer_size = size
        

        self._hidden_layers = nn.Sequential(*layers)
       
       # The logits layer: outputs the probability distribution
        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

        # The value output
        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )


    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_len):
        print("in forward")
        obs = input_dict["obs_flat"]["representation"]
        #self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._last_flat_in = obs
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features)
        # Apply the mask on the logits
        logits=logits-BIG_NUMBER*(1-input_dict["obs_flat"]["action_mask"])
        self._value = self._value_branch(self._features)

        print("*** the logits are ****", logits)
        print("*** the state is ****", state)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._value.squeeze(1)

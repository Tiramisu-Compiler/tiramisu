import argparse
from time import time
import numpy as np

import os
from Environment_sparse_enhanced import SearchSpaceSparseEnhanced
import ray.rllib.agents.ppo as ppo

from CustomModelV2Enhanced import TiramisuModelV2
import sys

import ray
from ray import tune

from ray.rllib.models.catalog import ModelCatalog


from ray.tune.registry import register_env


# Put the path to your tiramisu installation here
#tiramisu_path = '/scratch/nhh256/tiramisu/' 
tiramisu_path = '/home/narimane/tiramisu/'
os.environ['TIRAMISU_ROOT'] = tiramisu_path

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"]="1"

register_env("Tiramisu_env_v0", lambda a:SearchSpaceSparseEnhanced("../demo_test_enhanced_hyper.json", "../benchmarks_sources"))


ModelCatalog.register_custom_model("tiramisu_model_v1",TiramisuModelV2)



parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", default=0, type=int)
parser.add_argument("--training-iteration", default=1, type=int)
parser.add_argument("--ray-num-cpus", default=28, type=int)

args = parser.parse_args()
ray.init(num_cpus=args.ray_num_cpus)



agent = ppo.PPOTrainer(
    env="Tiramisu_env_v0",
    config={
        # "env": "Tiramisu_env_v1",
        "num_workers": args.num_workers,
        "batch_mode":"complete_episodes",
        "train_batch_size":1024,
        "sgd_minibatch_size": 256,
        "lr": 1e-4,
        "num_sgd_iter": 4,
        "explore": False,
        "framework":"torch",
        "_disable_preprocessor_api": True,
        "model": {
            "custom_model": "tiramisu_model_v1",
            "custom_model_config": {
                    "layer_sizes":[128, 1024, 1024, 128],
                    "drops":[0.225, 0.225, 0.225, 0.225]
                }
        },
    },
)



agent.restore("/home/narimane/Downloads/checkpoint_000045/checkpoint-45")

file_path="test_output.txt"
sys.stdout = open(file_path, "w+")

env= SearchSpaceSparseEnhanced("../demo_test_enhanced_hyper.json", "../benchmarks_sources")


observation, done = env.reset(), False

while not done:
    action = agent.compute_action(observation)
    observation, reward, done, _ = env.step(action)

    


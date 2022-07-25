import argparse
from unicodedata import name
from CustomModelV2Enhanced import TiramisuModelV2
import os, sys
from Environment_sparse_enhanced import SearchSpaceSparseEnhanced
import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env


# file_path="run_out.txt"
#sys.stdout = open(file_path, "w+")
#tiramisu_path = '/scratch/nhh256/tiramisu/' # Put the path to your tiramisu installation here
tiramisu_path='/home/narimane/tiramisu/'

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"]="1"
os.environ["RAY_ALLOW_SLOW_STORAGE"] = "1"

os.environ['TIRAMISU_ROOT'] = tiramisu_path

register_env("Tiramisu_env_v1", lambda a:SearchSpaceSparseEnhanced("../../../Desktop/PFE/new_dataset.json", "../../../Desktop/PFE/Dataset"))

f=open("output_results.txt", "w+")
f.write("")

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", default=0, type=int)
parser.add_argument("--training-iteration", default=1000, type=int)
parser.add_argument("--ray-num-cpus", default=6, type=int)
#parser.add_argument("--ray-num-gpus", default=1, type=int)
parser.add_argument("--checkpoint-freq", default=5, type=int)
args = parser.parse_args()
ray.init(num_cpus=args.ray_num_cpus)

ModelCatalog.register_custom_model("tiramisu_model_v1",TiramisuModelV2)

analysis=tune.run(
    "PPO",
    #local_dir="/scratch/nhh256/ray_results",
    name="Training_1000_sparse_enhanced_hyper",
    stop={"training_iteration": args.training_iteration},
    max_failures=0,
    checkpoint_freq=args.checkpoint_freq,
    config={
        "env": "Tiramisu_env_v1",
        "num_cpus_per_worker": 28,
        "num_workers": args.num_workers,
        "batch_mode":"complete_episodes",
        "train_batch_size":1024,
        "sgd_minibatch_size": 256,
        "lr": 1e-4,
        "num_sgd_iter":4,
        "framework":"torch",
        "_disable_preprocessor_api": True,
        "model": {
            "custom_model": "tiramisu_model_v1",
            "custom_model_config": {
                    "layer_sizes":[128, 1024, 1024, 128],
                    #"layer_sizes":[600, 350, 200, 7],
                    "drops":[0.225, 0.225, 0.225, 0.225]
                }
        },
    
    },
)


#Ray-RLlib script for training
import os, ray, argparse
from CustomModelEnhancedMult import TiramisuModelMult
from Environment_sparse_enhancedMult import SearchSpaceSparseEnhancedMult
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env

tiramisu_path = '/scratch/ne2128/tiramisu/' # Put the path to your tiramisu installation here
os.environ['TIRAMISU_ROOT'] = tiramisu_path

#The two environment variables below are set to 1 to avoid a Docker container error
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"]="1"
os.environ["RAY_ALLOW_SLOW_STORAGE"] = "1"

#Register the custom environment and the model
register_env("Tiramisu_env_v1", lambda a:SearchSpaceSparseEnhancedMult("../multicomp.json", "../Dataset_Multi"))
ModelCatalog.register_custom_model("tiramisu_model_v1",TiramisuModelMult)

#Configure the experiment
parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", default=3, type=int)
parser.add_argument("--training-iteration", default=1000, type=int)
parser.add_argument("--ray-num-cpus", default=112, type=int)
#parser.add_argument("--ray-num-gpus", default=1, type=int)
parser.add_argument("--checkpoint-freq", default=5, type=int)
args = parser.parse_args()
ray.init(num_cpus=args.ray_num_cpus)

analysis=tune.run(
    "PPO",
    local_dir="/scratch/ne2128/ray_results",
    name="Training_multi_enhanced",
    stop={"training_iteration": args.training_iteration},
    max_failures=0,
    checkpoint_freq=args.checkpoint_freq,
    config={
        "env": "Tiramisu_env_v1",
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
                    #'layer_sizes":[128, 1024, 1024, 128],
                    "layer_sizes":[600, 350, 200, 180],
                    "drops":[0.225, 0.225, 0.225, 0.225]
                }
        },
    
    },
)
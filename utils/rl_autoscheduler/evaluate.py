import ray.rllib.agents.ppo as ppo
import os
import json
# import hydra
import argparse
import ray
# from hydra.core.config_store import ConfigStore
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env

from rl_interface.environment import TiramisuScheduleEnvironment
from rl_interface.model import TiramisuModelMult
from utils.environment_variables import configure_env_variables
from utils.global_ray_variables import Actor, GlobalVarActor
from utils.rl_autoscheduler_config import (RLAutoSchedulerConfig,
                                           dict_to_config, parse_yaml_file,
                                           read_yaml_file)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=-1, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    return parser.parse_args()


# @hydra.main(config_path="config", config_name="config")
def main(config: RLAutoSchedulerConfig, checkpoint=None):
    if checkpoint is None: return
    configure_env_variables(config)
    best_checkpoint = os.path.join(config.ray.base_path, checkpoint)
    with ray.init(num_cpus=config.ray.ray_num_cpus):
        progs_list_registery = GlobalVarActor.remote(
            config.environment.programs_file,
            config.environment.dataset_path,
            num_workers=config.ray.num_workers)
        shared_variable_actor = Actor.remote(progs_list_registery)

        register_env(
            "Tiramisu_env_v1",
            lambda a: TiramisuScheduleEnvironment(config, shared_variable_actor
                                                  ),
        )
        ModelCatalog.register_custom_model("tiramisu_model_v1",
                                           TiramisuModelMult)

        agent = ppo.PPOTrainer(
            env="Tiramisu_env_v1",
            config={
                "num_workers": config.ray.num_workers,
                "batch_mode": "complete_episodes",
                "train_batch_size": 1024,
                "sgd_minibatch_size": 256,
                "lr": 1e-4,
                "num_sgd_iter": 4,
                "explore": False,
                "framework": "torch",
                "_disable_preprocessor_api": True,
                "model": {
                    "custom_model": "tiramisu_model_v1",
                    "custom_model_config": {
                        "layer_sizes": list(config.model.layer_sizes),
                        "drops": list(config.model.drops),
                    }
                },
            },
        )

        agent.restore(best_checkpoint)

        env = TiramisuScheduleEnvironment(config, shared_variable_actor)

        results = []
        while True:
            depth = 0
            result = dict()
            observation, done = env.reset(), False
            result["prog"] = env.prog.name
            while not done:
                depth = 0
                if depth > 14:
                    break
                try:
                    action = agent.compute_action(observation)
                    observation, reward, done, _ = env.step(action)
                except:
                    print("error", action, observation, reward, done)
                    continue
            result["schedule_str"] = env.schedule_object.schedule_str
            result["speedup"] = env.schedule_controller.speedup
            results.append(result)
            with open("results.json", "w+") as file:
                file.write(json.dumps(results))


if __name__ == "__main__":
    parsed_yaml_dict = parse_yaml_file(read_yaml_file("config.yaml"))
    config = dict_to_config(parsed_yaml_dict)
    args = get_arguments()
    if args.num_workers != -1:
        config.ray.num_workers = args.num_workers
    config.environment.programs_file = "./val.json"
    config.environment.dataset_path = "../benchmark"
    main(config, args.checkpoint)

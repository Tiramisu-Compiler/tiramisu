from dataclasses import dataclass, field
from typing import List
@dataclass
class ExperimentConfig:
    name: str = "base-model"
    base_path: str = "/home/user/recursive_lstm"


@dataclass
class DataGenerationConfig:
    train_dataset_file: str = "./datasets/dataset_batch550000-838143_train.pkl"
    valid_dataset_file: str = "./datasets/dataset_batch550000-838143_val.pkl"
    benchmark_dataset_file: str = "./datasets/benchmarks_mats1.pkl"
    dataset_name: str = "dataset"
    batch_size: int = 1024


@dataclass
class TrainConfig:
    log_file: str = "log.txt"
    lr: float = 0.001
    max_epochs: int = 1000


@dataclass
class TestConfig:
    datasets: List = field(default_factory=lambda: ["bench"])
    checkpoint: str = "best_model.pt"


@dataclass
class ModelConfig:
    comp_embed_layer_sizes: List = field(default_factory=lambda: [600, 350, 200, 180])
    drops: List = field(default_factory=lambda: [0.225, 0.225, 0.225, 0.225])
    input_size: int = 884


@dataclass
class WandbConfig:
    use_wandb: bool = True
    project: str = "recursive_lstm"


@dataclass
class RecursiveLSTMConfig:
    experiment: ExperimentConfig
    data_generation: DataGenerationConfig
    training: TrainConfig
    testing: TestConfig
    model: ModelConfig
    wandb: WandbConfig

    def __post_init__(self):
        if isinstance(self.experiment, dict):
            self.experiment = ExperimentConfig(**self.experiment)

    def __post_init__(self):
        if isinstance(self.data_generation, dict):
            self.data_generation = DataGenerationConfig(**self.data_generation)

    def __post_init__(self):
        if isinstance(self.training, dict):
            self.training = TrainConfig(**self.training)

    def __post_init__(self):
        if isinstance(self.testing, dict):
            self.testing = TestConfig(**self.testing)

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)

    def __post_init__(self):
        if isinstance(self.wandb, dict):
            self.wandb = WandbConfig(**self.wandb)

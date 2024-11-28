from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    random_seed: int = 42
    pretrained: str = "bert-base-uncased"
    npratio: int = 10
    history_size: int = 50
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # batch_size = 16 x 8 = 128
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_len: int = 128


cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainConfig)

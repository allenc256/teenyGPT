import configparser
import dataclasses
import torch

from .config import DatasetConfig, ModelConfig, TrainConfig
from .dataset import Datasets, create_datasets
from .model import Model, TransformerModel


def init_pytorch():
    """
    Initializes global pytorch settings.
    """
    # Use CUDA if available.
    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    # Use fp32.
    torch.set_default_dtype(torch.float32)


def init_from_config(config_file: str) -> tuple[Datasets, Model, TrainConfig]:
    """
    Utility method for initializing datasets, model, and training config from a
    config file.
    """
    cp = configparser.ConfigParser()
    cp.read(config_file)
    dataset_config = DatasetConfig.parse(cp["dataset"])
    datasets = create_datasets(dataset_config)
    train_config = TrainConfig.parse(cp["train"])
    model_config = ModelConfig.parse(cp["model"], datasets.encoder.vocab_size())

    # Create model.
    model = TransformerModel(model_config)

    # Dump config info.
    print()
    print("Configuration:")
    print()
    _pretty_print_dict(
        {
            "config file path": config_file,
            "dataset config": dataclasses.asdict(dataset_config),
            "model config": dataclasses.asdict(model_config),
            "training config": dataclasses.asdict(train_config),
            "training dataset size": f"{datasets.train.chunk_count} chunks",
            "validation dataset size": f"{datasets.val.chunk_count} chunks",
            "test dataset size": f"{datasets.test.chunk_count} chunks",
            "model size": f"{model.param_count()} parameters",
        }
    )
    print()

    return (datasets, model, train_config)


def _pretty_print_dict(d: dict, width: int = 30, indent: int = 0) -> None:
    for k, v in d.items():
        k = " " * indent + k
        if isinstance(v, dict):
            print(f"{k:<{width}}:")
            _pretty_print_dict(v, width, indent + 4)
        else:
            print(f"{k:<{width}}: {v}")

import argparse

import teenygpt

# Parse arguments.
parser = argparse.ArgumentParser(
    description="Trains a model from scratch.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--config",
    type=str,
    help="the config file containing dataset, model, and training parameters",
    default="config.ini",
)
args = parser.parse_args()

# Setup pytorch global settings.
teenygpt.init_pytorch()

# Load config & datasets.
datasets, model, train_config = teenygpt.init_from_config(args.config)

# Create model & run training.
trainer = teenygpt.Trainer(model, datasets, train_config)
trainer.train()

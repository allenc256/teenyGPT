import argparse

import torch

import teenygpt

# Parse arguments.
parser = argparse.ArgumentParser(
    description="Generates sampled continuations from an already trained model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--config",
    type=str,
    help="the config file containing dataset, model, and training parameters",
    default="config.ini",
)
parser.add_argument(
    "--split",
    type=str,
    choices=("train", "val", "test"),
    help="the dataset split to sample inputs from",
    default="test",
)
parser.add_argument(
    "--examples",
    type=int,
    help="the number of examples to generate",
    default=5,
)
parser.add_argument(
    "--length",
    type=int,
    help="the length (in tokens) of each generated example",
    default=512,
)
args = parser.parse_args()

# Initialization
datasets, model, train_config = teenygpt.init_from_config(args.config)

# Load model checkpoint.
if train_config.checkpoint_file is None:
    raise RuntimeError("checkpoint_file must be specified in config")
checkpoint = torch.load(train_config.checkpoint_file)
model.load_state_dict(checkpoint["model_state_dict"])

# Generate examples.
dataset = getattr(datasets, args.split)
inputs, _ = dataset.get_batch(args.examples, train_config.n_context)
generated = model.generate(inputs, args.length)

# Print examples.
for i in range(generated.size(0)):
    title = f"EXAMPLE {i+1}"
    print()
    print("-" * len(title))
    print(title)
    print("-" * len(title))
    print()
    print(datasets.encoder.decode(inputs[i].tolist()), end=" [[START]] ")
    print(datasets.encoder.decode(generated[i].tolist()))

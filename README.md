# teenyGPT

A small library which implements a Transformer from scratch using pytorch. This
project is only meant to serve as a demonstration only and is not optimized for
training large models on large datasets.

## Dependencies

Dependencies are listed in `requirements.txt` and may be installed via `pip`:

```
pip install -r requirements.txt
```

## Usage

This project includes a training script (`train.py`) that can be executed as so:

```
python train.py
```

All dataset, model, and training parameters used for training are specified in a
single config file (`config.ini`). The defaults provided will train a small 4
layer, 256 token context length model over `data/tiny_shakespeare.txt` (this
takes around ~1 minute or so on a RTX 4090).

This project also includes a generation script (`generate.py`) which can be used
to generate sample continuations:

```
% python generate.py -h
usage: generate.py [-h] [--config CONFIG] [--split {train,val,test}]
                   [--examples EXAMPLES] [--length LENGTH]

Generates sampled continuations from an already trained model.

options:
  -h, --help            show this help message and exit
  --config CONFIG       the config file containing dataset, model, and training
                        parameters (default: config.ini)
  --split {train,val,test}
                        the dataset split to sample inputs from (default: test)
  --examples EXAMPLES   the number of examples to generate (default: 5)
  --length LENGTH       the length (in tokens) of each generated example
                        (default: 512)
```

Some example output from the small trained model:

```
Rich'd! why march! but the noble-famious, if scor'd!
Be you will ent read a pooor suse
Unforcent of being for the breedings sold.

POLIXENES:
Didst thou, I disd
Do not battle. Apollo see concer'd from thee.
Then good my son, farewell; But so;
But spear behalf them seat by this in throne.

CAMILLO:
Who, let me too my lord?

JOMENIES:
Good my lords!

POLIXENES:
I think so!

PERCHIO:
Marry, so I know much my mother,
But ere to lordshing, spirits of my y lord.
```

## Model Details

The following design choices were made in the design of the model:

* Attention is computed using the pytorch function
  `scaled_dot_product_attention` to take advantage of pytorch-optimized kernels
  for attention.
* Feed-forward networks use the Swish GLU variant described in ["GLU Variants
  Improve Transformer"](https://arxiv.org/abs/2002.05202) as this appears to
  improve performance (and is what is used in Llama 2).
* We use pre-layer-norm rather than post-layer-norm as
  [research](https://arxiv.org/abs/2002.04745) suggests pre-LN is easier to
  train. In practice, many production-scale modules such as Llama 2 seem to
  favor pre-LN.
* The following positional encoding implementations are provided:
  * `ALIBI` - "Attention w/ Linear Biases" encoding scheme proposed in
    ["Train Short, Test Long: Attention with Linear Biases Enables Input Length
    Extrapolation"](https://arxiv.org/abs/2108.12409). Parameterless and purports
    to generalize
    well to arbitrary sequence lengths (e.g., inference on sequences longer
    than those encountered during training).
  * `SINUSOIDAL` - the original positional encoding proposed in
    ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). Also
    parameterless, but purportedly generalizes less well than ALiBi.
  * `LEARNED` - positional encoding via a learned embedding over position
    indices. Provides the best performance, but requires learning additional
    parameters and does not generalize to new sequence lengths.
  * `NONE` - no positional encoding.

## References

The following papers, codebases, and blog posts were referenced when building
this project.

* ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
* ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202)
* ["On Layer Normalization in the Transformer
  Architecture"](https://arxiv.org/abs/2002.04745)
* ["Train Short, Test Long: Attention with Linear Biases Enables Input Length
  Extrapolation"](https://arxiv.org/abs/2108.12409)
* [Llama 2](https://github.com/facebookresearch/llama)
* [nanoGPT](https://github.com/karpathy/nanoGPT) and
  [tinyshakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
  by Andrej Karpathy.
* [llama-from-scratch](https://github.com/bkitano/llama-from-scratch) by Brian Kitano

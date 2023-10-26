import configparser
import json
from dataclasses import dataclass
from enum import Enum
from typing import cast


@dataclass
class DatasetConfig:
    """
    Configuration for generating dataset splits.

    Attributes:
        input_file: File containing input text.
        input_file_encoding: Encoding of input text file.
        sentencepiece_model_file: SentencePiece model file to use for encoding text into
            tokens. If unspecified, a character-level encoding will be used instead of
            a subword encoding.
        chars_per_chunk: Chunk size, in characters, that the input text should be split
            into.
        fraction_train: The fraction of chunks that should be assigned to the train
            dataset split.
        fraction_val: The fraction of chunks that should be assigned to the validation
            dataset split.
        random_seed: The random seed (passed to ``random.Random()``) that should be
            used when deciding how to assign chunks to splits. When specified, this
            should guarantee determinism when generating dataset splits.
    """

    input_file: str
    input_file_encoding: str
    tokens_per_chunk: int
    fraction_train: float
    fraction_val: float
    sentencepiece_model_file: str | None = None
    random_seed: int | None = None

    @staticmethod
    def parse(section: configparser.SectionProxy) -> "DatasetConfig":
        """
        Parses config from a config section.
        """
        random_seed = section.get("random_seed")
        random_seed = int(random_seed) if random_seed is not None else None
        return DatasetConfig(
            input_file=section.get("input_file"),
            input_file_encoding=section.get("input_file_encoding"),
            tokens_per_chunk=int(section.get("tokens_per_chunk")),
            fraction_train=float(section.get("fraction_train")),
            fraction_val=float(section.get("fraction_val")),
            sentencepiece_model_file=section.get("sentencepiece_model_file"),
            random_seed=random_seed,
        )


class PositionalEncodingType(Enum):
    NONE = 0
    """No positional encoding. """

    ALIBI = 1
    """ALiBI encoding, as per https://arxiv.org/abs/2108.12409."""

    SINUSOIDAL = 2
    """Sinusoidal encoding, as per https://arxiv.org/abs/1706.03762."""

    LEARNED_EMBEDDING = 3
    """Learned positional encoding via a positional embedding."""

    LEARNED_SINUSOIDAL = 4
    """Learned positional encoding via sinusoids."""

    ROPE = 5
    """RoPE encoding, as per https://arxiv.org/abs/2104.09864."""


class FFNType(Enum):
    CLASSIC = 0
    """Classic encoding, as per https://arxiv.org/abs/1706.03762."""

    SWISH_GLU = 1
    """SwiGLU encoding, as per https://arxiv.org/abs/2002.05202."""


class LayerNormType(Enum):
    NONE = 0
    """No layer normalization."""

    CLASSIC = 1
    """Classic layer normalization."""

    RMS = 2
    """RMS layer normalization, as per https://arxiv.org/abs/1910.07467."""


@dataclass
class ModelConfig:
    """
    Configuration for a transformer model.

    Attributes:
        n_dim: The number of embedding dimensions. This must be evenly divisible by the
            number of attention heads (``n_attn_heads``).
        n_vocab: The maximum vocabulary size.
        n_attn_heads: The number of attention heads.
        n_attn_layers: The number attention layers.
        n_context_max: The maximum context length that the model can handle.
        p_dropout: The dropout probability.
        positional_encoding_type: The positional encoding type to use.
        ffn_type: Feed-forward network type to use.
        layer_norm_type: Type of layer norm to use.
    """

    n_dim: int
    n_vocab: int
    n_attn_heads: int
    n_attn_layers: int
    n_context_max: int
    p_dropout: float
    positional_encoding_type: PositionalEncodingType
    ffn_type: FFNType
    layer_norm_type: LayerNormType

    @staticmethod
    def parse(section: configparser.SectionProxy, n_vocab: int) -> "ModelConfig":
        """
        Parses config from a config section.
        """
        return ModelConfig(
            n_dim=int(section.get("n_dim")),
            n_vocab=n_vocab,
            n_attn_heads=int(section.get("n_attn_heads")),
            n_attn_layers=int(section.get("n_attn_layers")),
            n_context_max=int(section.get("n_context_max")),
            p_dropout=float(section.get("p_dropout")),
            positional_encoding_type=PositionalEncodingType[
                section.get("positional_encoding_type")
            ],
            ffn_type=FFNType[section.get("ffn_type")],
            layer_norm_type=LayerNormType[section.get("layer_norm_type")],
        )


@dataclass
class TrainConfig:
    """
    Configuration for training a transformer model.

    Attributes:
        n_iters: The number of training iterations.
        n_context: The context window length.
        n_batch: The batch size for the model.
        n_est_loss_iters: The number of training iterations to execute before
            estimating loss.
        n_est_loss_batches: The number of batches to use to estimate loss.
        lr_max: Maximum learning rate.
        lr_min: Minimum learning rate.
        weight_decay: Weight decay.
        betas: AdamW optimizer betas.
        checkpoint_file: File path specifying where checkpoints should be written to.
            If None, checkpoints will not be written.
    """

    n_iters: int
    n_batch: int
    n_context: int
    n_est_loss_iters: int
    n_est_loss_batches: int
    lr_max: float
    lr_min: float
    weight_decay: float
    betas: tuple[float, float]
    checkpoint_file: str | None

    @staticmethod
    def parse(section: configparser.SectionProxy) -> "TrainConfig":
        """
        Parses config from a config section.
        """
        return TrainConfig(
            n_iters=int(section.get("n_iters")),
            n_batch=int(section.get("n_batch")),
            n_context=int(section.get("n_context")),
            n_est_loss_iters=int(section.get("n_est_loss_iters")),
            n_est_loss_batches=int(section.get("n_est_loss_batches")),
            lr_max=float(section.get("lr_max")),
            lr_min=float(section.get("lr_min")),
            weight_decay=float(section.get("weight_decay")),
            betas=cast(tuple[float, float], tuple(json.loads(section.get("betas")))),
            checkpoint_file=section.get("checkpoint_file"),
        )

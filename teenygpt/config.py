from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    d_model: int = 64
    context_window: int = 16
    vocab_size: int = 512
    batch_size: int = 20
    attention_heads: int = 2
    attention_layers: int = 1
    dropout_p: float = 0.1
    rng: np.random.Generator = np.random.default_rng()

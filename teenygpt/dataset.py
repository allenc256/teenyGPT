import random
from enum import Enum

import numpy as np
import torch
import sentencepiece as spm  # type: ignore
from .config import Config


class Split(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class Encoder:
    def encode(self, input: str) -> list[int]:
        raise NotImplementedError()

    def decode(self, input: list[int]) -> str:
        raise NotImplementedError()

    def vocab_size(self) -> int:
        raise NotImplementedError()


class CharEncoder(Encoder):
    _char_to_index: dict[str, int]
    _index_to_char: list[str]

    def __init__(self, chars: str) -> None:
        super().__init__()
        self._index_to_char = sorted(set(chars))
        self._char_to_index = {
            ch: i for i, ch in enumerate(self._index_to_char)
        }

    def encode(self, input: str) -> list[int]:
        return [self._char_to_index[ch] for ch in input]

    def decode(self, input: list[int]) -> str:
        return "".join([self._index_to_char[i] for i in input])

    def vocab_size(self) -> int:
        return len(self._index_to_char)


class SentencePieceEncoder(Encoder):
    _processor: spm.SentencePieceProcessor
    _eos_id: int

    def __init__(self, processor: spm.SentencePieceProcessor) -> None:
        super().__init__()
        self._processor = processor
        self._eos_id = processor.eos_id()

    def encode(self, input: str) -> list[int]:
        lines = input.split("\n")
        result = []
        for i, encoded_line in enumerate(self._processor.encode(lines)):
            if i > 0:
                result.append(self._eos_id)
            result.extend(encoded_line)
        return result

    def decode(self, input: list[int]) -> str:
        return "\n".join(
            self._processor.decode(_split_list(input, self._eos_id))
        )


def _split_list(input: list[int], delimiter: int) -> list[list[int]]:
    offsets = [i for i, val in enumerate(input) if val == delimiter]
    last_offset = 0
    result = []
    for offset in offsets:
        result.append(input[last_offset:offset])
        last_offset = offset
    result.append(input[last_offset:])
    return result


class Dataset:
    split: Split
    _chunks: list[torch.Tensor]

    def __init__(self, split: Split, chunks: list[torch.Tensor]) -> None:
        self.split = split
        self._chunks = chunks

    def get_batches(self, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute probability to sample from each chunk, based on chunk length.
        p = np.array([len(chunk) for chunk in self._chunks], dtype=np.float64)
        p = np.maximum(p - config.context_window - 1, 0)
        if np.all(p <= 0):
            raise ValueError("no chunks are long enough to be sampled")
        p /= p.sum()

        # Sample from each chunk according to the distribution.
        xs = []
        ys = []
        for _ in range(config.batch_size):
            chunk = self._chunks[config.rng.choice(len(self._chunks), p=p)]
            x_start = config.rng.integers(
                0, len(chunk) - config.context_window - 1
            )
            x_stop = x_start + config.context_window
            y_start = x_start + 1
            y_stop = x_stop + 1
            xs.append(chunk[x_start:x_stop])
            ys.append(chunk[y_start:y_stop])

        return torch.stack(xs), torch.stack(ys)


def create_datasets(
    text: str,
    encoder: Encoder,
    chars_per_chunk: int = 10000,
    fraction_train: float = 0.8,
    fraction_val: float = 0.1,
) -> tuple[Dataset, Dataset, Dataset]:
    # Split lines into chunks.
    chunks = [
        torch.tensor(
            encoder.encode(text[i : i + chars_per_chunk]), dtype=torch.long
        )
        for i in range(0, len(text), chars_per_chunk)
    ]

    # Shuffle chunks.
    random.shuffle(chunks)

    # Split into train/val/test.
    chunks_train = chunks[: int(len(chunks) * fraction_train)]
    chunks_val = chunks[
        len(chunks_train) : len(chunks_train) + int(len(chunks) * fraction_val)
    ]
    chunks_test = chunks[len(chunks_train) + len(chunks_val) :]

    return (
        Dataset(Split.TRAIN, chunks_train),
        Dataset(Split.VALIDATION, chunks_val),
        Dataset(Split.TEST, chunks_test),
    )

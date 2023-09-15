import os

import torch
from tqdm import tqdm  # type: ignore

from .config import TrainConfig
from .dataset import Datasets
from .model import Model


class Trainer:
    """
    Class which encapsulates the training loop for a model.
    """

    # The model being trained.
    model: Model

    # The dataset splits to use to train and evaluate the model.
    datasets: Datasets

    # The training config.
    config: TrainConfig

    # The training optimizer.
    optimizer: torch.optim.Optimizer

    # The current iteration.
    iteration: int

    # Estimated training and validation losses.
    train_losses: list[float]
    val_losses: list[float]

    def __init__(
        self,
        model: Model,
        datasets: Datasets,
        config: TrainConfig,
        iteration: int = 0,
        train_losses: list[float] | None = None,
        val_losses: list[float] | None = None,
    ) -> None:
        self.model = model
        self.datasets = datasets
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters())
        self.iteration = iteration
        self.train_losses = train_losses or []
        self.val_losses = val_losses or []

    def train(self) -> None:
        """
        Executes the training loop.

        Checkpoints are saved automatically whenever validation loss improves. Progress
        during training is displayed via a tqdm progress bar.
        """
        progress = tqdm(range(self.iteration, self.config.n_iters))
        for i in progress:
            self.iteration = i

            # Grab next batch.
            xs, ys = self.datasets.train.get_batch(
                self.config.n_batch, self.config.n_context
            )

            # Zero out gradients.
            self.optimizer.zero_grad()

            # Forward pass.
            logits = self.model(xs)
            loss = self.model.loss(logits, ys)

            # Backward pass.
            loss.backward()
            self.optimizer.step()

            # Estimate loss and checkpoint if enough iterations have elapsed.
            if i % self.config.n_est_loss_iters == 0 or i == self.config.n_iters - 1:
                self._estimate_losses_and_checkpoint()
                progress.set_description(f"best_loss: {self._best_loss():0.2f}")

    def _best_loss(self) -> float:
        return min(self.val_losses) if len(self.val_losses) > 0 else torch.inf

    def _estimate_losses_and_checkpoint(self) -> None:
        train_loss = self.model.estimate_loss(self.datasets.train, self.config)
        val_loss = self.model.estimate_loss(self.datasets.val, self.config)

        # Save a checkpoint if validation loss has improved.
        if val_loss <= self._best_loss():
            self._save_checkpoint()

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def _save_checkpoint(self) -> None:
        if self.config.checkpoint_file is None:
            return

        # Make parent dirs if necessary.
        os.makedirs(os.path.dirname(self.config.checkpoint_file), exist_ok=True)

        torch.save(
            {"model_state_dict": self.model.state_dict()},
            self.config.checkpoint_file,
        )

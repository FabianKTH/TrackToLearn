import torch

from torch.nn import CosineSimilarity
from torch.nn.utils.rnn import PackedSequence
from torch.optim import Adam
from typing import Tuple


class Supervised(object):

    def __init__(
        self,
        policy,
        lr,
        device
    ):
        self.policy = policy
        self.lr = lr
        self.device = device
        self.cosine_loss = CosineSimilarity()
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

    def _update_actor(
        self,
        inputs,
        targets,
    ):
        """
        """

        total_actor_loss = 0

        outputs = self.policy.act(inputs)

        # Compute loss
        loss = -self.cosine_loss(outputs, targets)
        assert(torch.all(torch.isfinite(loss)))

        # Backprop
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        total_actor_loss += loss.mean()

        return total_actor_loss

    def update(
        self,
        data: Tuple[PackedSequence, PackedSequence],
        is_training: bool = False
    ) -> float:
        """Run a batch of data through the model and return the mean loss.

        Parameters
        ----------
        data : tuple of (PackedSequence, PackedSequence)
            The input and target batches of sequences.
        criterion : torch.nn.Module
            The loss function to apply on the model output.
        is_training : bool
            If True, record the computation graph and backprop through the
            model parameters.

        Returns
        -------
        mean_loss : float
            The mean loss of the provided batch
        """
        self.policy.train()

        packed_inputs, packed_targets = data

        actor_losses = self._update_actor(
            packed_inputs, packed_targets)

        return actor_losses.detach().cpu()

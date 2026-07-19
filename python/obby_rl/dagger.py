from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO


@dataclass(frozen=True)
class TakeoffOracle:
    # Calibrated in the eight-lane transport. Successful scripted takeoffs were
    # observed at 13.60, 15.42, and 17.26 studs; 19.14 was already too early.
    minimum_distance: float = 13.5
    maximum_distance: float = 17.5
    adapt_to_gap: bool = True

    def labels(self, observations: np.ndarray) -> np.ndarray:
        checked = np.asarray(observations, dtype=np.float32)
        if checked.ndim != 2 or checked.shape[1] != 22:
            raise ValueError(f"expected observations shaped (N, 22), got {checked.shape}")
        distance = np.linalg.norm(
            checked[:, 6:9] * np.asarray([64.0, 32.0, 64.0], dtype=np.float32), axis=1
        )
        gap = checked[:, 9] * 10.0
        has_gap = self.adapt_to_gap & (gap > 0)
        minimum_distance = np.where(
            has_gap, gap + (self.minimum_distance - 7.0), self.minimum_distance
        )
        maximum_distance = np.where(
            has_gap, gap + (self.maximum_distance - 7.0), self.maximum_distance
        )
        takeoff = (
            (checked[:, 4] > 0.5)
            & (distance >= minimum_distance)
            & (distance <= maximum_distance)
        )
        return np.where(takeoff, 1.0, -1.0).astype(np.float32)


def fit_jump_head(
    model: PPO,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("invalid jump-head training hyperparameters")
    checked_observations = np.asarray(observations, dtype=np.float32)
    checked_labels = np.asarray(labels, dtype=np.float32)
    if checked_observations.shape != (len(checked_labels), 22):
        raise ValueError("DAgger observation/label shapes do not match")
    policy = model.policy
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.action_net.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(policy.action_net.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    device = policy.device
    losses: list[float] = []
    policy.set_training_mode(True)
    for _ in range(epochs):
        permutation = generator.permutation(len(checked_labels))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            observation_tensor = torch.as_tensor(
                checked_observations[indices], dtype=torch.float32, device=device
            )
            label_tensor = torch.as_tensor(
                checked_labels[indices], dtype=torch.float32, device=device
            )
            distribution = policy.get_distribution(observation_tensor)
            base_distribution = distribution.distribution
            assert base_distribution is not None
            predicted_jump = base_distribution.mean[:, 3]
            loss = torch.nn.functional.mse_loss(predicted_jump, label_tensor)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        losses.append(epoch_loss / batches)
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.set_training_mode(False)
    return losses


def fit_action_head(
    model: PPO,
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    jump_loss_weight: float = 4.0,
) -> list[float]:
    """Fit all action means while preserving the shared Stage 1 representation."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or jump_loss_weight <= 0:
        raise ValueError("invalid action-head training hyperparameters")
    checked_observations = np.asarray(observations, dtype=np.float32)
    checked_labels = np.asarray(labels, dtype=np.float32)
    if checked_observations.ndim != 2 or checked_observations.shape[1] != 22:
        raise ValueError("DAgger observations must have shape (N, 22)")
    if checked_labels.shape != (len(checked_observations), 4):
        raise ValueError("DAgger action labels must have shape (N, 4)")
    policy = model.policy
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.action_net.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(policy.action_net.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    device = policy.device
    losses: list[float] = []
    policy.set_training_mode(True)
    for _ in range(epochs):
        permutation = generator.permutation(len(checked_labels))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            observation_tensor = torch.as_tensor(
                checked_observations[indices], dtype=torch.float32, device=device
            )
            label_tensor = torch.as_tensor(
                checked_labels[indices], dtype=torch.float32, device=device
            )
            distribution = policy.get_distribution(observation_tensor)
            base_distribution = distribution.distribution
            assert base_distribution is not None
            squared_error = torch.square(base_distribution.mean - label_tensor)
            action_weights = torch.as_tensor(
                [1.0, 1.0, 1.0, jump_loss_weight], dtype=torch.float32, device=device
            )
            loss = torch.mean(squared_error * action_weights)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        losses.append(epoch_loss / batches)
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.set_training_mode(False)
    return losses

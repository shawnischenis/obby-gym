from __future__ import annotations

import numpy as np
import torch
from torch import nn


class JumpFeasibilityModel(nn.Module):
    def __init__(self, observation_size: int = 48) -> None:
        super().__init__()
        # Exclude exact world/course positions and index/presence metadata that can
        # identify procedural seeds without causally determining jump physics.
        causal_indices = [*range(0, 22), *range(34, 44)]
        feature_mask = torch.zeros(observation_size)
        feature_mask[causal_indices] = 1
        self.register_buffer("feature_mask", feature_mask, persistent=False)
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations * self.feature_mask).squeeze(-1)


def fit_feasibility_model(
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> tuple[JumpFeasibilityModel, list[float]]:
    checked_observations = np.asarray(observations, dtype=np.float32)
    checked_labels = np.asarray(labels, dtype=np.float32)
    if checked_observations.ndim != 2 or checked_observations.shape[1] != 48:
        raise ValueError("feasibility observations must have shape (N, 48)")
    if checked_labels.shape != (len(checked_observations),):
        raise ValueError("feasibility labels must have shape (N,)")
    if float(np.min(checked_labels)) == float(np.max(checked_labels)):
        raise ValueError("feasibility labels require varying outcome probabilities")
    torch.manual_seed(seed)
    model = JumpFeasibilityModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    features = torch.as_tensor(checked_observations)
    targets = torch.as_tensor(checked_labels)
    negative_count = float(np.sum(1 - checked_labels))
    positive_count = float(np.sum(checked_labels))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative_count / positive_count))
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        logits = model(features)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, losses


def predict_probabilities(model: JumpFeasibilityModel, observations: np.ndarray) -> np.ndarray:
    features = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    with torch.no_grad():
        return torch.sigmoid(model(features)).numpy()

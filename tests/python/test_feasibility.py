from __future__ import annotations

import numpy as np
from obby_rl.feasibility import fit_feasibility_model, predict_probabilities


def test_feasibility_model_fits_separable_examples() -> None:
    observations = np.zeros((20, 48), dtype=np.float32)
    observations[:, 0] = np.linspace(-1, 1, 20)
    labels = (observations[:, 0] > 0).astype(np.float32)
    model, losses = fit_feasibility_model(observations, labels, epochs=100, learning_rate=0.01)
    probabilities = predict_probabilities(model, observations)
    assert losses[-1] < losses[0]
    assert np.mean((probabilities >= 0.5) == labels) >= 0.95

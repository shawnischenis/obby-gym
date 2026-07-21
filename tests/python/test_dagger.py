from __future__ import annotations

import numpy as np
from obby_rl.dagger import TakeoffOracle


def test_takeoff_oracle_labels_grounded_distance_window() -> None:
    observations = np.zeros((3, 22), dtype=np.float32)
    observations[:, 4] = 1
    observations[:, 8] = np.asarray([19 / 64, 16 / 64, 10 / 64])
    observations[2, 4] = 0
    labels = TakeoffOracle().labels(observations)
    np.testing.assert_array_equal(labels, [-1, 1, -1])


def test_takeoff_oracle_uses_vector_calibrated_boundaries() -> None:
    observations = np.zeros((4, 22), dtype=np.float32)
    observations[:, 4] = 1
    observations[:, 8] = np.asarray([13.4, 13.5, 17.5, 17.6]) / 64
    labels = TakeoffOracle().labels(observations)
    np.testing.assert_array_equal(labels, [-1, 1, 1, -1])


def test_takeoff_oracle_shifts_window_with_gap_feature() -> None:
    observations = np.zeros((4, 22), dtype=np.float32)
    observations[:, 4] = 1
    observations[:, 9] = np.asarray([0.5, 0.5, 1.0, 1.0])
    observations[:, 8] = np.asarray([15.4, 15.6, 16.4, 16.6]) / 64
    labels = TakeoffOracle().labels(observations)
    np.testing.assert_array_equal(labels, [1, -1, -1, 1])


def test_takeoff_oracle_never_jumps_on_non_jump_sentinel() -> None:
    observation = np.zeros((1, 22), dtype=np.float32)
    observation[0, 4] = 1
    observation[0, 8] = 16 / 64
    observation[0, 9] = -1
    np.testing.assert_array_equal(TakeoffOracle().labels(observation), [-1])

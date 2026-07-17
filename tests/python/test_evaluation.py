from __future__ import annotations

import numpy as np
import pytest
from obby_rl.env import RobloxObbyEnv
from obby_rl.evaluation import evaluate_seeds
from obby_rl.testing import FakeTransport
from obby_rl.training import make_fixed_course_env


class ZeroPolicy:
    def predict(
        self,
        observation: np.ndarray,
        state: object = None,
        episode_start: object = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        return np.zeros(4, dtype=np.float32), None


def test_evaluation_reports_completion_metrics() -> None:
    env = make_fixed_course_env(RobloxObbyEnv(FakeTransport()), course_seed=4, max_episode_steps=20)
    results = evaluate_seeds(ZeroPolicy(), env, [4, 4])
    assert results["episode_count"] == 2
    assert results["completion_rate"] == 1.0
    assert results["mean_length"] == 10
    assert results["mean_return"] == pytest.approx(1.0)
    env.close()

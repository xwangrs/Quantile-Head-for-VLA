#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces


@pytest.fixture(scope="module")
def libero_env_class():
    """Import the wrapper without requiring the optional LIBERO package."""
    try:
        return importlib.import_module("lerobot.envs.libero").LiberoEnv
    except ModuleNotFoundError as exc:
        if exc.name != "libero":
            raise

    libero_package = ModuleType("libero")
    libero_module = ModuleType("libero.libero")
    libero_envs_module = ModuleType("libero.libero.envs")
    libero_module.benchmark = SimpleNamespace(Benchmark=object)
    libero_module.get_libero_path = lambda _: ""
    libero_envs_module.OffScreenRenderEnv = object
    libero_package.libero = libero_module

    modules = {
        "libero": libero_package,
        "libero.libero": libero_module,
        "libero.libero.envs": libero_envs_module,
    }
    original_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        return importlib.import_module("lerobot.envs.libero").LiberoEnv
    finally:
        sys.modules.pop("lerobot.envs.libero", None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _FakeController:
    use_delta = True


class _FakeLiberoBackend:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.set_init_state_calls: list[int] = []
        self.robots = [SimpleNamespace(controller=_FakeController())]

    @staticmethod
    def _observation(value: int) -> dict[str, np.ndarray]:
        return {"agentview_image": np.full((2, 2, 3), value, dtype=np.uint8)}

    def seed(self, seed: int | None) -> None:
        del seed

    def reset(self) -> dict[str, np.ndarray]:
        self.reset_calls += 1
        return self._observation(10 + self.reset_calls)

    def set_init_state(self, init_state: np.ndarray) -> dict[str, np.ndarray]:
        self.set_init_state_calls.append(int(init_state.item()))
        return self._observation(20 + self.set_init_state_calls[-1])

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        del action
        return self._observation(99), 1.0, False, {}

    @staticmethod
    def check_success() -> bool:
        return True

    def close(self) -> None:
        pass


def _make_test_env(libero_env_class):
    env = object.__new__(libero_env_class)
    gym.Env.__init__(env)
    env._env = _FakeLiberoBackend()
    env.task = "test_task"
    env.task_description = "test task"
    env.task_id = 0
    env.camera_name = ["agentview_image"]
    env.camera_name_mapping = {"agentview_image": "image"}
    env.obs_type = "pixels"
    env.init_states = True
    env._init_states = np.array([[0], [1], [2]], dtype=np.int64)
    env.init_state_id = 0
    env._reset_stride = 1
    env.num_steps_wait = 0
    env.control_mode = "relative"
    env._max_episode_steps = 10
    env.observation_space = spaces.Dict(
        {"pixels": spaces.Dict({"image": spaces.Box(low=0, high=255, shape=(2, 2, 3), dtype=np.uint8)})}
    )
    env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    return env


def test_terminal_step_does_not_reset_or_replace_terminal_observation(libero_env_class):
    env = _make_test_env(libero_env_class)
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))

    assert env._env.reset_calls == 1
    assert env.init_state_id == 1
    assert np.all(observation["pixels"]["image"] == 99)
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["is_success"]


def test_vector_env_performs_exactly_one_next_step_autoreset(libero_env_class):
    vector_env = gym.vector.SyncVectorEnv([lambda: _make_test_env(libero_env_class)])
    try:
        vector_env.reset(seed=[123])
        backend = vector_env.envs[0]._env
        assert backend.reset_calls == 1
        assert backend.set_init_state_calls == [0]

        observation, _, terminated, truncated, info = vector_env.step(np.zeros((1, 7), dtype=np.float32))

        assert terminated.tolist() == [True]
        assert truncated.tolist() == [False]
        assert info["is_success"].tolist() == [True]
        assert np.all(observation["pixels"]["image"] == 99)
        assert backend.reset_calls == 1
        assert backend.set_init_state_calls == [0]

        _, _, terminated, truncated, info = vector_env.step(np.zeros((1, 7), dtype=np.float32))

        assert terminated.tolist() == [False]
        assert truncated.tolist() == [False]
        assert info["is_success"].tolist() == [False]
        assert backend.reset_calls == 2
        assert backend.set_init_state_calls == [0, 1]
    finally:
        vector_env.close()

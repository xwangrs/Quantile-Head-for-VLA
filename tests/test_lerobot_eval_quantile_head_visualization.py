from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lerobot.scripts.lerobot_eval as lerobot_eval
from lerobot.policies.quantile_head.modeling_quantile_head import QuantileHeadPytorch
from lerobot.scripts.lerobot_eval import (
    _extract_executed_action_episodes,
    compute_executed_action_statistics,
    sample_quantile_head_action_distribution,
    save_executed_action_statistics,
    save_quantile_head_action_distribution,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from tests.quantile_head_test_utils import make_quantile_head_config_stub  # noqa: E402


class _FakeRegressionModel(QuantileHeadPytorch):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.register_buffer("quantile_levels", torch.tensor([0.1, 0.5, 0.9]), persistent=False)
        self.regression_calls = 0
        self.sampler_shapes = []

    def regress_quantiles_from_zero_query(self, images, image_masks, tokens, masks, time):
        del images, image_masks, masks
        self.regression_calls += 1
        assert tokens.shape == (1, 4)
        assert time.shape == (1,)
        return torch.arange(1 * 3 * 4 * 3, dtype=torch.float32).reshape(1, 3, 4, 3) / 10

    def interpolate_marginal_quantiles(self, quantile_knots, marginal_levels):
        self.sampler_shapes.append(tuple(quantile_knots.shape))
        return super().interpolate_marginal_quantiles(quantile_knots, marginal_levels)


class _FakeQuantileHeadPolicy(torch.nn.Module):
    def __init__(self, inference_mode="median"):
        super().__init__()
        self.config = make_quantile_head_config_stub(
            type="quantile_head",
            action_generation_mode="regression",
            output_features={ACTION: SimpleNamespace(shape=(2,))},
            chunk_size=3,
            n_action_steps=2,
            action_feature_names=None,
            marginal_quantile_inference_mode=inference_mode,
            marginal_quantile_sampling_temperature=0.35,
            marginal_quantile_sampling_min_level=0.1,
            marginal_quantile_sampling_max_level=0.9,
            marginal_quantile_sampling_ar_coefficient=0.9,
            marginal_quantile_sampling_seed=1000,
            marginal_quantile_sampling_exclude_joints=["gripper"],
        )
        self.model = _FakeRegressionModel()
        self._marginal_quantile_sampling_generator = torch.Generator(device="cpu")
        self._marginal_quantile_sampling_generator.manual_seed(1000)
        self._marginal_quantile_sampling_latent = torch.tensor([[0.25, -0.5]])

    def _preprocess_images(self, batch):
        del batch
        return [], []


class _FakePeftWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def get_base_model(self):
        return self.policy


def _batch(batch_size=1):
    return {
        OBS_LANGUAGE_TOKENS: torch.zeros(batch_size, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 4, dtype=torch.bool),
    }


def test_sample_quantile_head_action_distribution_samples_marginals_and_preserves_rng():
    policy = _FakeQuantileHeadPolicy()
    wrapper = _FakePeftWrapper(policy)
    postprocessor_shapes = []

    def postprocessor(action):
        postprocessor_shapes.append(tuple(action.shape))
        return action * 10 + 1

    torch.manual_seed(777)
    expected_next_random_values = torch.rand(4)
    torch.manual_seed(777)
    policy_sampler_state = policy._marginal_quantile_sampling_generator.get_state().clone()
    policy_latent = policy._marginal_quantile_sampling_latent.clone()

    distribution = sample_quantile_head_action_distribution(
        wrapper,
        _batch(),
        postprocessor,
        lambda transition: transition,
        n_samples=5,
        sample_batch_size=2,
        seed=123,
    )

    actual_next_random_values = torch.rand(4)
    torch.testing.assert_close(actual_next_random_values, expected_next_random_values)
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), policy_sampler_state)
    assert torch.equal(policy._marginal_quantile_sampling_latent, policy_latent)
    assert policy.model.regression_calls == 1
    assert policy.model.sampler_shapes == [
        (2, 2, 2, 3),
        (2, 2, 2, 3),
        (1, 2, 2, 3),
    ]
    assert postprocessor_shapes == [
        (2, 2, 2),
        (2, 2, 2),
        (1, 2, 2),
        (3, 2, 2),
        (1, 2, 2),
    ]
    assert distribution["marginal_samples_normalized"].shape == (5, 2, 2)
    assert distribution["marginal_samples_env"].shape == (5, 2, 2)
    assert distribution["median_actions_normalized"].shape == (2, 2)
    assert distribution["median_actions_env"].shape == (2, 2)
    assert distribution["quantile_knots_normalized"].shape == (2, 2, 3)
    assert distribution["quantile_knots_env"].shape == (2, 2, 3)
    torch.testing.assert_close(
        distribution["marginal_samples_env"],
        distribution["marginal_samples_normalized"] * 10 + 1,
    )
    torch.testing.assert_close(
        distribution["quantile_knots_env"],
        distribution["quantile_knots_normalized"] * 10 + 1,
    )
    torch.testing.assert_close(
        distribution["median_actions_normalized"],
        distribution["quantile_knots_normalized"][..., 1],
    )
    torch.testing.assert_close(
        distribution["median_actions_env"],
        distribution["quantile_knots_env"][..., 1],
    )
    assert not torch.allclose(
        distribution["marginal_samples_normalized"],
        distribution["median_actions_normalized"].expand(5, -1, -1),
    )
    torch.testing.assert_close(distribution["quantile_levels"], torch.tensor([0.1, 0.5, 0.9]))
    assert distribution["action_names"] == ["action_0", "action_1"]


@pytest.mark.parametrize(
    ("n_samples", "sample_batch_size", "match"),
    [
        (0, 2, "n_samples must be positive"),
        (2, 0, "sample_batch_size must be positive"),
    ],
)
def test_sample_quantile_head_action_distribution_validates_sample_counts(
    n_samples, sample_batch_size, match
):
    with pytest.raises(ValueError, match=match):
        sample_quantile_head_action_distribution(
            _FakeQuantileHeadPolicy(),
            _batch(),
            lambda action: action,
            lambda transition: transition,
            n_samples=n_samples,
            sample_batch_size=sample_batch_size,
        )


def test_sample_quantile_head_action_distribution_requires_one_observation():
    with pytest.raises(ValueError, match="eval.batch_size=1"):
        sample_quantile_head_action_distribution(
            _FakeQuantileHeadPolicy(),
            _batch(batch_size=2),
            lambda action: action,
            lambda transition: transition,
            n_samples=2,
        )


def test_sample_quantile_head_action_distribution_requires_regression_mode():
    policy = _FakeQuantileHeadPolicy()
    policy.config.action_generation_mode = "flow_matching"

    with pytest.raises(ValueError, match="action_generation_mode='regression'"):
        sample_quantile_head_action_distribution(
            policy,
            _batch(),
            lambda action: action,
            lambda transition: transition,
            n_samples=2,
        )


@pytest.mark.parametrize("wrapped", [False, True])
def test_sample_quantile_head_action_distribution_rejects_pure_regression_before_quantile_access(wrapped):
    policy = torch.nn.Module()
    policy.config = make_quantile_head_config_stub(
        type="quantile_head", action_generation_mode="regression", use_pure_regression_head=True
    )
    wrapped_policy = _FakePeftWrapper(policy) if wrapped else policy

    with pytest.raises(ValueError, match="requires a quantile head.*use_pure_regression_head=True") as error:
        sample_quantile_head_action_distribution(
            wrapped_policy,
            _batch(),
            lambda action: action,
            lambda transition: transition,
            n_samples=2,
        )

    assert lerobot_eval.QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV in str(error.value)


@pytest.mark.parametrize("decoder", ["policy", "independent_rank", "gaussian_temporal"])
def test_save_quantile_head_action_distribution_writes_safe_outputs(tmp_path, decoder):
    pytest.importorskip("matplotlib")
    policy = _FakeQuantileHeadPolicy(inference_mode="temporal_sample")
    lerobot_eval.configure_quantile_head_eval_decoder(policy, decoder)

    output_paths = save_quantile_head_action_distribution(
        policy,
        _batch(),
        lambda action: action,
        lambda transition: transition,
        tmp_path,
        "../task name",
        n_samples=5,
        horizon_step=1,
    )

    assert output_paths.keys() == {"histogram", "cdf", "data"}
    for path in output_paths.values():
        assert path.is_relative_to(tmp_path)
        assert path.is_file()
        assert path.stat().st_size > 0

    with np.load(output_paths["data"]) as saved:
        assert saved["marginal_samples_env"].shape == (5, 2, 2)
        assert saved["median_actions_env"].shape == (2, 2)
        assert saved["quantile_knots_env"].shape == (2, 2, 3)
        np.testing.assert_allclose(saved["median_actions_env"], saved["quantile_knots_env"][..., 1])
        np.testing.assert_allclose(saved["quantile_levels"], [0.1, 0.5, 0.9])
        assert saved["policy_decoder"].item() == ("temporal_sample" if decoder == "policy" else decoder)
        assert saved["diagnostic_sampling"].item() == "independent_marginals"
        assert saved["diagnostic_distribution"].item() == "learned_quantiles"
        assert saved["sampling_temperature"].item() == pytest.approx(0.35)
        assert saved["sampling_min_level"].item() == pytest.approx(0.1)
        assert saved["sampling_max_level"].item() == pytest.approx(0.9)
        assert saved["sampling_ar_coefficient"].item() == pytest.approx(0.9)
        assert saved["sampling_seed"].item() == 1000
        assert saved["sampling_exclude_joints"].tolist() == ["gripper"]
        assert not any(key.startswith("copula") for key in saved.files)


def test_compute_executed_action_statistics_differences_each_episode_separately():
    episode_actions = [
        np.asarray(
            [
                [0.0, 0.0, 5.0],
                [1.0, 1.0, 5.0],
                [3.0, 3.0, 5.0],
            ]
        ),
        np.asarray(
            [
                [100.0, -100.0, 5.0],
                [103.0, -97.0, 5.0],
                [107.0, -93.0, 5.0],
            ]
        ),
    ]

    statistics = compute_executed_action_statistics(
        episode_actions,
        ["action_0", "action_1", "constant"],
    )

    np.testing.assert_array_equal(statistics["episode_lengths"], [3, 3])
    np.testing.assert_allclose(
        statistics["delta_actions_env"],
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0],
        ],
    )
    np.testing.assert_allclose(statistics["delta_correlation"][:2, :2], np.ones((2, 2)))
    assert np.isnan(statistics["delta_correlation"][2]).all()
    assert np.isnan(statistics["delta_correlation"][:, 2]).all()


def test_extract_executed_action_episodes_respects_done_and_partial_batch():
    actions = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    done_indices = torch.tensor([1, 3])

    episodes = _extract_executed_action_episodes(actions, done_indices, max_episodes=1)

    assert len(episodes) == 1
    np.testing.assert_array_equal(episodes[0], actions[0, :2].numpy())


def test_eval_policy_collects_action_stats_when_quantile_head_viz_is_enabled(monkeypatch, tmp_path):
    policy = _FakeQuantileHeadPolicy()
    env = SimpleNamespace(num_envs=1)
    rollout_results = [
        {
            ACTION: torch.tensor([[[0.0, 0.0], [1.0, 2.0], [999.0, 999.0]]]),
            "reward": torch.zeros(1, 3),
            "success": torch.zeros(1, 3, dtype=torch.bool),
            "done": torch.tensor([[False, True, True]]),
            "task_name": "pick the cup",
        },
        {
            ACTION: torch.tensor([[[10.0, 10.0], [11.0, 12.0], [13.0, 13.0]]]),
            "reward": torch.zeros(1, 3),
            "success": torch.ones(1, 3, dtype=torch.bool),
            "done": torch.tensor([[False, False, True]]),
            "task_name": "pick the cup",
        },
    ]
    saved = {}

    def fake_rollout(**kwargs):
        del kwargs
        return rollout_results.pop(0)

    def fake_save(episode_actions, action_names, output_dir, task_name):
        saved["episode_actions"] = episode_actions
        saved["action_names"] = action_names
        saved["output_dir"] = output_dir
        saved["task_name"] = task_name
        return {"r_delta_csv": output_dir / "R_delta.csv"}

    monkeypatch.setenv(lerobot_eval.QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(lerobot_eval, "PreTrainedPolicy", torch.nn.Module)
    monkeypatch.setattr(lerobot_eval, "rollout", fake_rollout)
    monkeypatch.setattr(lerobot_eval, "save_executed_action_statistics", fake_save)

    info = lerobot_eval.eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=2,
    )

    assert [len(actions) for actions in saved["episode_actions"]] == [2, 3]
    np.testing.assert_array_equal(saved["episode_actions"][0], [[0.0, 0.0], [1.0, 2.0]])
    assert saved["action_names"] == ["action_0", "action_1"]
    assert saved["output_dir"] == tmp_path
    assert saved["task_name"] == "pick the cup"
    assert info["action_stats_paths"] == {"r_delta_csv": str(tmp_path / "R_delta.csv")}


def test_save_executed_action_statistics_writes_numeric_and_visual_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    episode_actions = [
        np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 3.0]]),
        np.asarray([[20.0, -20.0], [23.0, -19.0], [27.0, -16.0]]),
    ]

    output_paths = save_executed_action_statistics(
        episode_actions,
        ["dx", "dy"],
        tmp_path,
        "../task name",
    )

    assert output_paths.keys() == {
        "data",
        "r_action_csv",
        "r_delta_csv",
        "r_action_plot",
        "r_delta_plot",
    }
    for path in output_paths.values():
        assert path.is_relative_to(tmp_path)
        assert path.is_file()
        assert path.stat().st_size > 0

    with np.load(output_paths["data"]) as saved:
        np.testing.assert_array_equal(saved["episode_lengths"], [3, 3])
        assert saved["actions_env"].shape == (6, 2)
        assert saved["delta_actions_env"].shape == (4, 2)
        np.testing.assert_allclose(
            saved["r_delta"],
            compute_executed_action_statistics(episode_actions, ["dx", "dy"])["delta_correlation"],
        )
        assert saved["action_names"].tolist() == ["dx", "dy"]
        assert saved["scale"].item() == "environment"
        assert saved["correlation"].item() == "pearson"

import json
import logging
from copy import deepcopy
from math import atan, pi
from statistics import NormalDist
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.policies.quantile_head.modeling_quantile_head import (  # noqa: E402
    QuantileHeadPolicy,
    QuantileHeadPytorch,
)
from lerobot.scripts.quantile_head_eval_decoders import (  # noqa: E402
    configure_quantile_head_eval_decoder,
    describe_quantile_head_quantile_sampler,
    export_quantile_head_quantile_width_calibration,
)
from tests.quantile_head_test_utils import make_quantile_head_config_stub  # noqa: E402


class _QuantileModel(QuantileHeadPytorch):
    def __init__(self, levels):
        torch.nn.Module.__init__(self)
        self.register_buffer("quantile_levels", torch.tensor(levels), persistent=False)
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))


class _PeftWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def get_base_model(self):
        return self.policy


def _make_policy(
    *,
    seed=31,
    temperature=0.7,
    rho=0.9,
    min_level=0.1,
    max_level=0.9,
    levels=(0.05, 0.275, 0.5, 0.725, 0.95),
    exclude_joints=(),
    feature_names=None,
    inference_mode="median",
    action_dim=3,
):
    policy = QuantileHeadPolicy.__new__(QuantileHeadPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = make_quantile_head_config_stub(
        type="quantile_head",
        action_generation_mode="regression",
        output_features={"action": SimpleNamespace(shape=(action_dim,))},
        n_action_steps=3,
        marginal_quantile_inference_mode=inference_mode,
        marginal_quantile_sampling_temperature=temperature,
        marginal_quantile_sampling_min_level=min_level,
        marginal_quantile_sampling_max_level=max_level,
        marginal_quantile_sampling_ar_coefficient=rho,
        marginal_quantile_sampling_seed=seed,
        marginal_quantile_sampling_exclude_joints=list(exclude_joints),
        action_feature_names=feature_names,
    )
    policy.model = _QuantileModel(levels)
    policy._marginal_quantile_sampling_generator = torch.Generator(device="cpu")
    policy._marginal_quantile_sampling_generator.manual_seed(seed)
    policy.reset()
    return policy


def _identity_knots(policy, *, batch=2, horizon=4, action_dim=3, dtype=torch.float32):
    return policy.model.quantile_levels.to(dtype).view(1, 1, 1, -1).expand(batch, horizon, action_dim, -1)


def test_policy_decoder_is_a_true_noop():
    policy = _make_policy()
    original_decoder = policy._decode_regression_quantiles
    original_config = deepcopy(vars(policy.config))
    sampler_state = policy._marginal_quantile_sampling_generator.get_state().clone()

    assert configure_quantile_head_eval_decoder(policy, "policy") is None

    assert policy._decode_regression_quantiles == original_decoder
    assert "_decode_regression_quantiles" not in vars(policy)
    assert vars(policy.config) == original_config
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_state)


@pytest.mark.parametrize("exclude_joints", [[], ["gripper"]])
@pytest.mark.parametrize(
    ("decoder", "inference_mode", "temperature", "sampling_enabled"),
    [
        ("policy", "median", 0.35, False),
        ("policy", "temporal_sample", 0.35, True),
        ("independent_rank", "median", 0.35, True),
        ("gaussian_temporal", "median", 0.35, True),
        ("gaussian_temporal", "temporal_sample", 0.0, False),
    ],
)
def test_sampler_metadata_records_resolved_seven_dimensional_mask_without_mutation(
    exclude_joints, decoder, inference_mode, temperature, sampling_enabled
):
    policy = _make_policy(
        action_dim=7,
        feature_names=["actions"],
        exclude_joints=exclude_joints,
        inference_mode=inference_mode,
        temperature=temperature,
    )
    wrapper = _PeftWrapper(policy)
    configure_quantile_head_eval_decoder(wrapper, decoder)
    original_config = deepcopy(vars(policy.config))
    original_attributes = set(vars(policy))
    sampler_state = policy._marginal_quantile_sampling_generator.get_state().clone()
    global_state = torch.random.get_rng_state().clone()

    metadata = describe_quantile_head_quantile_sampler(wrapper)

    excluded = [6] if exclude_joints else []
    assert metadata["action_dim"] == 7
    assert metadata["action_feature_names"] == ["actions"]
    assert metadata["coordinate_names_complete"] is False
    assert metadata["sampling_exclude_joints"] == exclude_joints
    assert metadata["sampling_exclusion_mask"] == [False] * 6 + [bool(exclude_joints)]
    assert metadata["excluded_action_indices"] == excluded
    assert metadata["sampling_enabled"] is sampling_enabled
    assert metadata["sampled_action_indices"] == (
        [index for index in range(7) if index not in excluded] if sampling_enabled else []
    )
    assert metadata["decoder"] == (inference_mode if decoder == "policy" else decoder)
    assert metadata["soft_prompt_len"] == 0
    assert metadata["soft_prompt_location"] is None
    assert "num_soft_prompt_tokens" not in metadata
    assert vars(policy.config) == original_config
    assert set(vars(policy)) == original_attributes
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert policy._marginal_quantile_sampling_latent is None
    assert policy._last_marginal_quantile_levels is None


@pytest.mark.parametrize("soft_prompt_len", [0, 16])
def test_sampler_metadata_identifies_action_suffix_prompt_without_mutation(soft_prompt_len):
    policy = _make_policy()
    policy.config.soft_prompt_len = soft_prompt_len
    original_config = deepcopy(vars(policy.config))
    original_attributes = set(vars(policy))
    sampler_state = policy._marginal_quantile_sampling_generator.get_state().clone()
    global_state = torch.random.get_rng_state().clone()

    metadata = describe_quantile_head_quantile_sampler(policy)

    assert metadata["soft_prompt_len"] == soft_prompt_len
    assert metadata["soft_prompt_location"] == ("action_suffix" if soft_prompt_len else None)
    assert "num_soft_prompt_tokens" not in metadata
    assert vars(policy.config) == original_config
    assert set(vars(policy)) == original_attributes
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert policy._marginal_quantile_sampling_latent is None
    assert policy._last_marginal_quantile_levels is None


@pytest.mark.parametrize("policy_type", [None, "act", "quantile_head_flow"])
def test_sampler_metadata_skips_other_policy_types(policy_type):
    policy = torch.nn.Module()
    if policy_type is not None:
        policy.config = make_quantile_head_config_stub(
            type="quantile_head" if policy_type == "quantile_head_flow" else policy_type,
            action_generation_mode="flow_matching",
        )
    assert describe_quantile_head_quantile_sampler(policy) is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_pure_regression_policy_has_no_quantile_sampler_and_default_decoder_is_noop(wrapped):
    policy = torch.nn.Module()
    policy.config = make_quantile_head_config_stub(
        type="quantile_head", action_generation_mode="regression", use_pure_regression_head=True
    )
    original_attributes = set(vars(policy))
    original_config = deepcopy(vars(policy.config))
    wrapped_policy = _PeftWrapper(policy) if wrapped else policy

    assert configure_quantile_head_eval_decoder(wrapped_policy, "policy") is None
    assert describe_quantile_head_quantile_sampler(wrapped_policy) is None

    assert set(vars(policy)) == original_attributes
    assert vars(policy.config) == original_config
    assert not hasattr(policy, "_marginal_quantile_sampling_generator")


@pytest.mark.parametrize("decoder", ["independent_rank", "gaussian_temporal"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_quantile_ablation_rejects_pure_regression_before_accessing_quantile_fields(decoder, wrapped):
    policy = torch.nn.Module()
    policy.config = make_quantile_head_config_stub(
        type="quantile_head", action_generation_mode="regression", use_pure_regression_head=True
    )
    wrapped_policy = _PeftWrapper(policy) if wrapped else policy

    with pytest.raises(ValueError, match="requires a quantile head.*use_pure_regression_head=True"):
        configure_quantile_head_eval_decoder(wrapped_policy, decoder)

    assert not hasattr(policy, "_decode_regression_quantiles")
    assert not hasattr(policy, "_quantile_head_eval_decoder_name")


@pytest.mark.parametrize("decoder", ["independent_rank", "gaussian_temporal"])
def test_eval_decoder_overrides_median_on_peft_base_without_changing_checkpoint(decoder):
    policy = _make_policy()
    wrapper = _PeftWrapper(policy)
    original_config = deepcopy(vars(policy.config))
    original_weights = {key: value.clone() for key, value in policy.state_dict().items()}
    knots = _identity_knots(policy)

    metadata = configure_quantile_head_eval_decoder(wrapper, decoder)
    actions = policy._decode_regression_quantiles(knots)

    assert isinstance(metadata, dict)
    assert policy._quantile_head_eval_decoder_name == decoder
    assert "_decode_regression_quantiles" in vars(policy)
    assert "_decode_regression_quantiles" not in vars(wrapper)
    assert not torch.equal(actions, knots[..., 2])
    assert vars(policy.config) == original_config
    assert policy.state_dict().keys() == original_weights.keys()
    for key, value in policy.state_dict().items():
        assert torch.equal(value, original_weights[key])


def test_independent_rank_resamples_every_horizon_ignores_rho_and_preserves_global_rng():
    first = _make_policy(rho=0.0, min_level=0.2, max_level=0.85)
    second = _make_policy(rho=1.0, min_level=0.2, max_level=0.85)
    latent = torch.tensor([[0.2, -0.3, 0.4], [0.5, -0.6, 0.7]])
    first._marginal_quantile_sampling_latent = latent.clone()
    second._marginal_quantile_sampling_latent = latent.clone()
    configure_quantile_head_eval_decoder(first, "independent_rank")
    configure_quantile_head_eval_decoder(second, "independent_rank")
    knots = _identity_knots(first)
    reference_rng = torch.Generator(device="cpu").manual_seed(31)
    global_rng_state = torch.random.get_rng_state().clone()
    previous_actions = None

    for _ in range(2):
        # This reference also fixes the draw shape/order, not just the marginal bounds.
        gaussian_draws = torch.randn((2, 4, 3), generator=reference_rng)
        centered_rank = torch.erf(0.7 * gaussian_draws / 2**0.5)
        expected = torch.where(centered_rank < 0, 0.5 + centered_rank * 0.3, 0.5 + centered_rank * 0.35)
        actions = first._decode_regression_quantiles(knots)

        torch.testing.assert_close(actions, expected)
        torch.testing.assert_close(actions, second._decode_regression_quantiles(knots))
        assert not torch.equal(actions[:, 0], actions[:, 1])
        assert torch.all((actions >= 0.2) & (actions <= 0.85))
        assert torch.equal(first._marginal_quantile_sampling_latent, latent)
        assert torch.equal(second._marginal_quantile_sampling_latent, latent)
        assert torch.equal(first._marginal_quantile_sampling_generator.get_state(), reference_rng.get_state())
        if previous_actions is not None:
            assert not torch.equal(actions, previous_actions)
        previous_actions = actions

    assert torch.equal(torch.random.get_rng_state(), global_rng_state)


@pytest.mark.parametrize("decoder", ["independent_rank", "gaussian_temporal"])
def test_zero_temperature_returns_exact_median_without_rng_or_ar_changes(decoder):
    policy = _make_policy(
        temperature=0.0, min_level=0.2, max_level=0.95, feature_names=["actions"], exclude_joints=["gripper"]
    )
    policy._marginal_quantile_sampling_latent = torch.arange(7, dtype=torch.float32).view(1, 7)
    latent = policy._marginal_quantile_sampling_latent.clone()
    knots = torch.tensor([-4.0, -2.0, 0.125, 2.0, 8.0], dtype=torch.bfloat16)
    knots = knots.view(1, 1, 1, 5).expand(1, 4, 7, 5)
    configure_quantile_head_eval_decoder(policy, decoder)
    sampler_state = policy._marginal_quantile_sampling_generator.get_state().clone()
    global_state = torch.random.get_rng_state().clone()

    actions = policy._decode_regression_quantiles(knots)

    assert torch.equal(actions, knots[..., 2])
    assert actions.dtype == knots.dtype
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert torch.equal(policy._marginal_quantile_sampling_latent, latent)
    assert torch.all(policy._last_marginal_quantile_levels == 0.5)


@pytest.mark.parametrize("decoder", ["independent_rank", "gaussian_temporal"])
@pytest.mark.parametrize(
    ("feature_names", "exclude_joints", "excluded_index"),
    [
        (None, ["gripper"], 6),
        ([], ["gripper"], 6),
        (["actions"], ["gripper"], 6),
        (["x", "y", "robot.gripper", "z", "rx", "ry", "rz"], ["gripper"], 2),
        (["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"], ["gripper"], None),
        (["actions"], [], None),
    ],
)
def test_eval_decoders_respect_effective_gripper_exclusion(
    decoder, feature_names, exclude_joints, excluded_index
):
    policy = _make_policy(exclude_joints=exclude_joints, feature_names=feature_names, rho=0.7)
    sample_all = _make_policy(feature_names=feature_names, rho=0.7)
    configure_quantile_head_eval_decoder(policy, decoder)
    configure_quantile_head_eval_decoder(sample_all, decoder)
    knots = _identity_knots(policy, action_dim=7, dtype=torch.bfloat16)
    knots = knots.clone()
    knots[..., -1, :] += 1.5

    for _ in range(2):
        actions = policy._decode_regression_quantiles(knots)
        all_actions = sample_all._decode_regression_quantiles(knots)
        levels = policy._last_marginal_quantile_levels

        for index in range(7):
            if index == excluded_index:
                assert torch.equal(actions[..., index], knots[..., index, 2])
                assert torch.all(levels[..., index] == 0.5)
                assert not torch.equal(all_actions[..., index], actions[..., index])
            else:
                assert torch.equal(actions[..., index], all_actions[..., index])
                assert not torch.equal(actions[..., index], knots[..., index, 2])
                assert torch.any(levels[..., index] != 0.5)
        assert torch.equal(
            policy._marginal_quantile_sampling_generator.get_state(),
            sample_all._marginal_quantile_sampling_generator.get_state(),
        )
    baseline_actions = policy.model.interpolate_marginal_quantiles(knots, torch.full_like(knots[..., 0], 0.5))
    assert actions.dtype == baseline_actions.dtype
    assert actions.device == knots.device


def test_gaussian_uses_interpolated_central_width_and_only_one_ar_helper_call():
    policy = _make_policy()
    calls = []
    ranks = torch.tensor([[0.2, 0.5, 0.8]])

    def fixed_ranks(self, batch_size, action_dim):
        del self
        calls.append((batch_size, action_dim))
        return ranks.clone()

    policy._sample_temporally_coherent_quantile_levels = MethodType(fixed_ranks, policy)
    configure_quantile_head_eval_decoder(policy, "gaussian_temporal")
    knots = torch.tensor([-4.0, -2.0, 1.0, 2.0, 8.0]).view(1, 1, 1, 5).expand(1, 2, 3, 5).clone()
    knots[:, 1] = knots[:, 1] * 2 + 3

    actions = policy._decode_regression_quantiles(knots)

    lower = -4 + (0.1 - 0.05) / (0.275 - 0.05) * 2
    upper = 2 + (0.9 - 0.725) / (0.95 - 0.725) * 6
    scale = (upper - lower) / (2 * NormalDist().inv_cdf(0.9))
    standardized = torch.tensor([NormalDist().inv_cdf(rank) for rank in (0.2, 0.5, 0.8)])
    expected_first = 1 + scale * standardized
    torch.testing.assert_close(actions[0, 0], expected_first)
    torch.testing.assert_close(actions[0, 1], expected_first * 2 + 3)
    assert calls == [(1, 3)]


def test_gaussian_preserves_original_ar_and_rng_sequence_across_replans_and_reset():
    reference = _make_policy(inference_mode="temporal_sample", rho=0.75)
    gaussian = _make_policy(inference_mode="temporal_sample", rho=0.75)
    configure_quantile_head_eval_decoder(gaussian, "gaussian_temporal")
    knots = _identity_knots(gaussian)
    global_rng = torch.random.get_rng_state().clone()

    for should_reset in (False, False, True):
        if should_reset:
            generator_before_reset = gaussian._marginal_quantile_sampling_generator.get_state().clone()
            reference.reset()
            gaussian.reset()
            assert gaussian._marginal_quantile_sampling_latent is None
            assert gaussian._last_marginal_quantile_levels is None
            assert torch.equal(
                gaussian._marginal_quantile_sampling_generator.get_state(), generator_before_reset
            )
            assert gaussian._quantile_head_eval_decoder_name == "gaussian_temporal"
            assert "_decode_regression_quantiles" in vars(gaussian)

        reference._decode_regression_quantiles(knots)
        gaussian._decode_regression_quantiles(knots)

        assert torch.equal(
            gaussian._marginal_quantile_sampling_latent, reference._marginal_quantile_sampling_latent
        )
        assert torch.equal(gaussian._last_marginal_quantile_levels, reference._last_marginal_quantile_levels)
        assert torch.equal(
            gaussian._marginal_quantile_sampling_generator.get_state(),
            reference._marginal_quantile_sampling_generator.get_state(),
        )

    assert torch.equal(torch.random.get_rng_state(), global_rng)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("all_collapsed", [False, True])
def test_gaussian_zero_width_and_endpoint_ranks_stay_finite_without_changing_rng_budget(dtype, all_collapsed):
    policy = _make_policy(temperature=1e6, min_level=0.0, max_level=1.0)
    reference = _make_policy(temperature=1e6, min_level=0.0, max_level=1.0)
    configure_quantile_head_eval_decoder(policy, "gaussian_temporal")
    knots = _identity_knots(policy, dtype=dtype).clone()
    knots[..., 0, :] = 0.25
    if all_collapsed:
        knots.fill_(0.25)
    expected_ranks = reference._sample_temporally_coherent_quantile_levels(2, 3)
    assert torch.any((expected_ranks == 0) | (expected_ranks == 1))

    actions = policy._decode_regression_quantiles(knots)

    assert torch.isfinite(actions).all()
    baseline_actions = policy.model.interpolate_marginal_quantiles(knots, torch.full_like(knots[..., 0], 0.5))
    assert actions.dtype == baseline_actions.dtype
    assert torch.equal(actions[..., 0], knots[..., 0, 2])
    assert torch.equal(
        policy._marginal_quantile_sampling_generator.get_state(),
        reference._marginal_quantile_sampling_generator.get_state(),
    )


@pytest.mark.parametrize("decoder", ["independent_rank", "gaussian_temporal"])
@pytest.mark.parametrize("invalid_policy", ["other_policy", "flow_matching"])
def test_eval_decoders_reject_unsupported_policy_types(decoder, invalid_policy):
    policy = _make_policy()
    if invalid_policy == "other_policy":
        policy.config.type = "act"
    else:
        policy.config.action_generation_mode = "flow_matching"

    with pytest.raises(ValueError):
        configure_quantile_head_eval_decoder(policy, decoder)


def test_eval_decoder_rejects_unknown_name():
    with pytest.raises(ValueError):
        configure_quantile_head_eval_decoder(_make_policy(), "unknown")


@pytest.mark.parametrize("levels", [(0.2, 0.5, 0.95), (0.05, 0.5, 0.8)])
def test_gaussian_rejects_quantile_grids_without_central_interval_coverage(levels):
    with pytest.raises(ValueError):
        configure_quantile_head_eval_decoder(_make_policy(levels=levels), "gaussian_temporal")


@pytest.mark.parametrize("decoder", ["independent_rank", "gaussian_temporal"])
@pytest.mark.parametrize("temperature", [-1.0, float("nan"), float("inf")])
def test_eval_decoders_reject_invalid_temperature(decoder, temperature):
    with pytest.raises(ValueError):
        configure_quantile_head_eval_decoder(_make_policy(temperature=temperature), decoder)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_independent_rank_matches_original_decoder_for_single_step_chunks(dtype):
    independent = _make_policy(rho=0.0)
    reference = _make_policy(rho=0.0, inference_mode="temporal_sample")
    configure_quantile_head_eval_decoder(independent, "independent_rank")
    knots = _identity_knots(reference, horizon=1, dtype=dtype)

    for _ in range(2):
        torch.testing.assert_close(
            independent._decode_regression_quantiles(knots),
            reference._decode_regression_quantiles(knots),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("decoder", ["policy", "independent_rank", "gaussian_temporal"])
def test_cli_decoder_is_applied_before_evaluation_and_recorded(monkeypatch, tmp_path, caplog, decoder):
    import draccus

    import lerobot.scripts.lerobot_eval as evaluation
    from lerobot.configs.eval import EvalPipelineConfig

    monkeypatch.setattr("sys.argv", ["lerobot-eval"])
    monkeypatch.delenv(evaluation.QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV, raising=False)
    cfg = draccus.parse(
        EvalPipelineConfig,
        args=[
            "--env.type=libero",
            "--policy.type=quantile_head",
            "--policy.device=cpu",
            "--policy.use_amp=false",
            f"--output_dir={tmp_path}",
            f"--quantile_head_quantile_decoder={decoder}",
        ],
    )
    policy = _make_policy(inference_mode="median", feature_names=["actions"], exclude_joints=["gripper"])
    policy.config.device = "cpu"
    knots = _identity_knots(policy)
    evaluated = []
    closed = []

    def evaluate(**kwargs):
        assert kwargs["policy"] is policy
        assert not policy.training
        actions = policy._decode_regression_quantiles(knots)
        if decoder == "policy":
            assert torch.equal(actions, knots[..., 2])
        else:
            assert policy._quantile_head_eval_decoder_name == decoder
            assert not torch.equal(actions, knots[..., 2])
        evaluated.append(True)
        return {"overall": {"pc_success": 50.0}}

    monkeypatch.setattr(evaluation, "make_env", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "make_policy", lambda **kwargs: policy)
    monkeypatch.setattr(evaluation, "make_pre_post_processors", lambda **kwargs: (None, None))
    monkeypatch.setattr(evaluation, "make_env_pre_post_processors", lambda **kwargs: (None, None))
    monkeypatch.setattr(evaluation, "set_seed", lambda seed: None)
    monkeypatch.setattr(evaluation, "eval_policy_all", evaluate)
    monkeypatch.setattr(evaluation, "close_envs", lambda envs: closed.append(envs))

    with caplog.at_level(logging.INFO):
        evaluation.eval_main(cfg)

    saved = json.loads((tmp_path / "eval_info.json").read_text())
    assert evaluated == [True]
    assert closed == [{}]
    assert saved["overall"]["pc_success"] == 50.0
    sampler = saved["quantile_head_quantile_sampler"]
    assert sampler["action_dim"] == 3
    assert sampler["sampling_exclusion_mask"] == [False, False, True]
    assert sampler["excluded_action_indices"] == [2]
    assert sampler["sampled_action_indices"] == ([] if decoder == "policy" else [0, 1])
    assert sampler["sampling_enabled"] is (decoder != "policy")
    assert sampler["decoder"] == ("median" if decoder == "policy" else decoder)
    assert any(
        "QuantileHead resolved quantile sampler:" in record.getMessage()
        and "'sampling_exclusion_mask': [False, False, True]" in record.getMessage()
        for record in caplog.records
    )
    if decoder == "policy":
        assert "quantile_head_quantile_ablation" not in saved
    else:
        metadata = saved["quantile_head_quantile_ablation"]
        assert metadata["decoder"] == decoder
        assert metadata["policy_inference_mode"] == "median"
        assert metadata["effective_ar_coefficient"] == (0.9 if decoder == "gaussian_temporal" else None)


@pytest.mark.parametrize("decoder", ["independent_rank", "gaussian_temporal"])
def test_eval_rejects_parallel_tasks_before_creating_environments(monkeypatch, tmp_path, decoder):
    import lerobot.scripts.lerobot_eval as evaluation
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.configs import LiberoEnv

    monkeypatch.setattr("sys.argv", ["lerobot-eval"])
    cfg = EvalPipelineConfig(
        env=LiberoEnv(max_parallel_tasks=2), output_dir=tmp_path, quantile_head_quantile_decoder=decoder
    )
    with pytest.raises(ValueError, match="max_parallel_tasks=1"):
        evaluation.eval_main(cfg)


def test_plain_evaluation_accepts_pure_regression_without_quantile_attributes(monkeypatch, tmp_path):
    import draccus

    import lerobot.scripts.lerobot_eval as evaluation
    from lerobot.configs.eval import EvalPipelineConfig

    monkeypatch.setattr("sys.argv", ["lerobot-eval"])
    monkeypatch.delenv(evaluation.QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV, raising=False)
    cfg = draccus.parse(
        EvalPipelineConfig,
        args=[
            "--env.type=libero",
            "--policy.type=quantile_head",
            "--policy.device=cpu",
            "--policy.use_amp=false",
            f"--output_dir={tmp_path}",
        ],
    )
    policy = torch.nn.Module()
    policy.config = make_quantile_head_config_stub(
        type="quantile_head", action_generation_mode="regression", use_pure_regression_head=True, device="cpu"
    )
    evaluated = []
    closed = []

    def evaluate(**kwargs):
        assert kwargs["policy"] is policy
        assert not policy.training
        evaluated.append(True)
        return {"overall": {"pc_success": 50.0}}

    monkeypatch.setattr(evaluation, "make_env", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "make_policy", lambda **kwargs: policy)
    monkeypatch.setattr(evaluation, "make_pre_post_processors", lambda **kwargs: (None, None))
    monkeypatch.setattr(evaluation, "make_env_pre_post_processors", lambda **kwargs: (None, None))
    monkeypatch.setattr(evaluation, "set_seed", lambda seed: None)
    monkeypatch.setattr(evaluation, "eval_policy_all", evaluate)
    monkeypatch.setattr(evaluation, "close_envs", lambda envs: closed.append(envs))

    evaluation.eval_main(cfg)

    saved = json.loads((tmp_path / "eval_info.json").read_text())
    assert evaluated == [True]
    assert closed == [{}]
    assert saved["overall"]["pc_success"] == 50.0
    assert "quantile_head_quantile_sampler" not in saved
    assert "quantile_head_quantile_ablation" not in saved


@pytest.mark.parametrize("temperature, bounds", [(0.35, (0.1, 0.9)), (0.35, (0.2, 0.8)), (1.0, (0.1, 0.9))])
def test_quantile_expectation_integrates_the_tempered_rank_distribution_without_sampling(temperature, bounds):
    policies = [
        _make_policy(
            seed=seed,
            temperature=temperature,
            rho=rho,
            min_level=bounds[0],
            max_level=bounds[1],
            levels=(0.0, 0.25, 0.5, 0.75, 1.0),
            feature_names=["x", "y", "gripper"],
            exclude_joints=["gripper"],
        )
        for seed, rho in ((7, 0.0), (999, 0.9))
    ]
    # Q(t)=t below the median and Q(t)=.5+3(t-.5) above it.
    # Its exact expectation is .5+w*2/pi*atan(T), where w is the rank half-width.
    # Taking the median or treating T=.35 ranks as uniform is incorrect.
    knots = torch.tensor([0.0, 0.25, 0.5, 1.25, 2.0]).view(1, 1, 1, 5).expand(2, 4, 3, 5)
    expected = 0.5 + (bounds[1] - 0.5) * 2 / pi * atan(temperature)
    global_state = torch.random.get_rng_state().clone()
    decoded = []

    for policy in policies:
        policy._marginal_quantile_sampling_latent = torch.full((2, 3), 0.25)
        latent = policy._marginal_quantile_sampling_latent.clone()
        sampler_state = policy._marginal_quantile_sampling_generator.get_state().clone()
        configure_quantile_head_eval_decoder(
            _PeftWrapper(policy), "quantile_expectation", expectation_nodes=128
        )

        actions = policy._decode_regression_quantiles(knots)

        torch.testing.assert_close(
            actions[..., :2], torch.full_like(actions[..., :2], expected), atol=0.0015, rtol=0
        )
        assert torch.equal(actions[..., 2], knots[..., 2, 2])
        assert torch.equal(actions, policy._decode_regression_quantiles(knots))
        assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_state)
        assert torch.equal(policy._marginal_quantile_sampling_latent, latent)
        metadata = describe_quantile_head_quantile_sampler(policy)
        assert metadata["sampling_enabled"] is False
        assert metadata["sampled_action_indices"] == []
        decoded.append(actions)

    assert torch.equal(decoded[0], decoded[1])
    assert torch.equal(torch.random.get_rng_state(), global_state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("temperature", [0.0, 0.35])
def test_quantile_expectation_preserves_constant_quantiles_and_output_dtype(dtype, temperature):
    policy = _make_policy(temperature=temperature)
    configure_quantile_head_eval_decoder(policy, "quantile_expectation")
    knots = _identity_knots(policy, dtype=dtype).clone()
    knots[..., 0, :] = 0.25
    reference = policy.model.interpolate_marginal_quantiles(knots, torch.full_like(knots[..., 0], 0.5))

    actions = policy._decode_regression_quantiles(knots)

    assert actions.device == knots.device
    assert actions.dtype == (knots.dtype if temperature == 0 else reference.dtype)
    assert torch.isfinite(actions).all()
    torch.testing.assert_close(actions[..., 0], knots[..., 0, 2].to(actions.dtype))
    torch.testing.assert_close(actions, reference.to(actions.dtype), atol=0.003, rtol=0)


@pytest.mark.parametrize("nodes", [0, -1, 1.5, True])
def test_quantile_expectation_rejects_invalid_integration_node_counts(nodes):
    with pytest.raises((TypeError, ValueError)):
        configure_quantile_head_eval_decoder(_make_policy(), "quantile_expectation", expectation_nodes=nodes)


def _write_fixed_widths(tmp_path, widths):
    path = tmp_path / "fixed_widths.json"
    path.write_text(json.dumps({"widths": widths}), encoding="utf-8")
    return path


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_fixed_width_removes_state_scale_but_keeps_asymmetric_shape_and_exclusions(tmp_path, dtype):
    path = _write_fixed_widths(tmp_path, [[0.8, 1.2, 4.0], [1.6, 0.4, 4.0]])
    policy = _make_policy(
        levels=(0.0, 0.25, 0.5, 0.75, 1.0),
        feature_names=["x", "y", "gripper"],
        exclude_joints=["gripper"],
    )
    calls = []

    def fixed_ranks(self, batch_size, action_dim):
        del self
        calls.append((batch_size, action_dim))
        return torch.tensor([[0.3, 0.7, 0.8]]).expand(batch_size, -1).clone()

    policy._sample_temporally_coherent_quantile_levels = MethodType(fixed_ranks, policy)
    configure_quantile_head_eval_decoder(policy, "fixed_width", fixed_width_path=path)
    centers = torch.tensor([0.0, 10.0], dtype=dtype).view(2, 1, 1).expand(2, 2, 3)
    shape = torch.tensor([-0.5, -0.25, 0.0, 0.75, 1.5], dtype=dtype)
    state_scales = torch.tensor([1.0, 4.0], dtype=dtype).view(2, 1, 1, 1)
    knots = centers[..., None] + state_scales * shape
    knots = knots.expand(2, 2, 3, 5).clone()
    knots[1, 1, 1, :] = centers[1, 1, 1]

    actions = policy._decode_regression_quantiles(knots)

    # Equal-distance ranks around .5 have a 1:3 displacement, not a Gaussian shape.
    offsets = torch.tensor([[-0.1, 0.45, 0.0], [-0.2, 0.15, 0.0]])
    expected = centers.float() + offsets
    expected[1, 1, 1] = centers[1, 1, 1]
    tolerance = 0.04 if dtype == torch.bfloat16 else 0.006 if dtype == torch.float16 else 0.00001
    torch.testing.assert_close(actions.float(), expected, atol=tolerance, rtol=0)
    assert torch.equal(actions[..., 2], knots[..., 2, 2])
    assert actions[1, 1, 1] == centers[1, 1, 1]
    reference = policy.model.interpolate_marginal_quantiles(knots, torch.full_like(knots[..., 0], 0.5))
    assert actions.dtype == reference.dtype
    assert actions.device == knots.device
    assert calls == [(2, 3)]


@pytest.mark.parametrize("temperature", [0.0, 0.35])
def test_fixed_width_broadcast_matches_original_sampler_and_preserves_rng_across_reset(tmp_path, temperature):
    path = _write_fixed_widths(tmp_path, [0.8, 0.8, 0.8])
    reference = _make_policy(temperature=temperature, inference_mode="temporal_sample", rho=0.75)
    policy = _make_policy(temperature=temperature, inference_mode="temporal_sample", rho=0.75)
    configure_quantile_head_eval_decoder(policy, "fixed_width", fixed_width_path=path)
    knots = _identity_knots(policy).clone()
    knots[..., 0, :] = 0.25
    global_state = torch.random.get_rng_state().clone()

    for should_reset in (False, False, True):
        if should_reset:
            reference.reset()
            policy.reset()
        torch.testing.assert_close(
            policy._decode_regression_quantiles(knots), reference._decode_regression_quantiles(knots)
        )
        assert torch.equal(policy._last_marginal_quantile_levels, reference._last_marginal_quantile_levels)
        if temperature == 0:
            assert policy._marginal_quantile_sampling_latent is None
        else:
            assert torch.equal(
                policy._marginal_quantile_sampling_latent, reference._marginal_quantile_sampling_latent
            )
        assert torch.equal(
            policy._marginal_quantile_sampling_generator.get_state(),
            reference._marginal_quantile_sampling_generator.get_state(),
        )

    assert torch.equal(torch.random.get_rng_state(), global_state)


@pytest.mark.parametrize(
    "widths",
    [
        [],
        0.8,
        [0.8, -1.0, 0.8],
        [0.8, float("nan"), 0.8],
        [0.8, float("inf"), 0.8],
        [0.8, 0.8],
        [[0.8, 0.8, 0.8]] * 3,
        [[0.8, 0.8, 0.8], [0.8]],
        [[[0.8, 0.8, 0.8]]],
    ],
)
def test_fixed_width_rejects_invalid_values_and_incompatible_shapes(tmp_path, widths):
    path = _write_fixed_widths(tmp_path, widths)
    policy = _make_policy()
    with pytest.raises((TypeError, ValueError)):
        configure_quantile_head_eval_decoder(policy, "fixed_width", fixed_width_path=path)
        policy._decode_regression_quantiles(_identity_knots(policy))


def test_fixed_width_requires_a_width_calibration_file(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        configure_quantile_head_eval_decoder(_make_policy(), "fixed_width")
    path = tmp_path / "missing_widths_key.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        configure_quantile_head_eval_decoder(_make_policy(), "fixed_width", fixed_width_path=path)


def test_width_calibration_weights_observations_equally_and_round_trips_to_fixed_width(tmp_path):
    policy = _make_policy(levels=(0.1, 0.3, 0.5, 0.7, 0.9), inference_mode="temporal_sample")
    wrapper = _PeftWrapper(policy)
    configure_quantile_head_eval_decoder(wrapper, "calibrate_width")
    batches = [
        torch.tensor([[[0.0, 3.0, 4.0], [1.0, 2.0, 6.0]], [[4.0, 3.0, 8.0], [5.0, 0.0, 6.0]]]),
        torch.tensor([[[8.0, 3.0, 12.0], [7.0, 8.0, 6.0]]]),
    ]
    shape = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5])
    sampler_state = policy._marginal_quantile_sampling_generator.get_state().clone()
    global_state = torch.random.get_rng_state().clone()

    for widths in batches:
        policy.reset()
        knots = 2.0 + widths[..., None] * shape
        assert torch.equal(policy._decode_regression_quantiles(knots), knots[..., 2])

    output_path = tmp_path / "calibration.json"
    metadata = export_quantile_head_quantile_width_calibration(wrapper, output_path)
    assert isinstance(metadata, dict)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    expected_widths = torch.cat(batches, dim=0).square().mean(dim=0).sqrt()
    torch.testing.assert_close(torch.tensor(saved["widths"]), expected_widths)
    assert describe_quantile_head_quantile_sampler(wrapper)["sampling_enabled"] is False
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert policy._marginal_quantile_sampling_latent is None

    restored = _make_policy(levels=(0.1, 0.3, 0.5, 0.7, 0.9))

    def fixed_ranks(self, batch_size, action_dim):
        del self, action_dim
        return torch.tensor([[0.1, 0.5, 0.9]]).expand(batch_size, -1).clone()

    restored._sample_temporally_coherent_quantile_levels = MethodType(fixed_ranks, restored)
    configure_quantile_head_eval_decoder(restored, "fixed_width", fixed_width_path=output_path)
    knots = 2.0 + batches[0][..., None] * shape
    expected = (2.0 + expected_widths * torch.tensor([-0.5, 0.0, 0.5])).expand(2, -1, -1).clone()
    expected[batches[0] == 0] = 2.0
    torch.testing.assert_close(restored._decode_regression_quantiles(knots), expected)


def test_width_calibration_cannot_export_before_observing_quantiles(tmp_path):
    policy = _make_policy()
    configure_quantile_head_eval_decoder(policy, "calibrate_width")
    with pytest.raises(ValueError):
        export_quantile_head_quantile_width_calibration(policy, tmp_path / "empty_calibration.json")

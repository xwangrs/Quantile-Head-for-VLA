"""Validate the public policy name and the format of newly saved checkpoints."""

import json

import pytest

torch = pytest.importorskip("torch")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig  # noqa: E402
from lerobot.policies.quantile_head.processor_quantile_head import (  # noqa: E402
    QuantileHeadPrepareStateTokenizerProcessorStep,
)
from lerobot.processor import PolicyProcessorPipeline, ProcessorStepRegistry  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402


def test_quantile_head_config_saves_and_restores_canonical_policy_type(tmp_path):
    config = QuantileHeadConfig(device="cpu", soft_prompt_len=7)
    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["type"] == "quantile_head"
    assert not any(name.startswith("copula_") for name in saved)
    assert "num_soft_prompt_tokens" not in saved
    assert "marginal_quantile_gap_barrier_strength" not in saved

    restored = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(restored, QuantileHeadConfig)
    assert restored.type == "quantile_head"
    assert restored.soft_prompt_len == 7


def test_pi05_is_not_registered_as_a_policy_alias():
    with pytest.raises(KeyError, match="pi05"):
        PreTrainedConfig.get_choice_class("pi05")


@pytest.mark.parametrize(
    "removed_field",
    [
        "marginal_quantile_gap_barrier_strength",
        "num_soft_prompt_tokens",
        "copula_rank",
        "copula_loss_weight",
        "copula_factor_scale",
        "copula_min_diagonal",
        "copula_backprop_to_action_model",
        "copula_deterministic_inference",
        "copula_shared_sampling_temperature",
        "copula_independent_sampling_temperature",
    ],
)
def test_removed_checkpoint_fields_are_rejected(removed_field):
    with pytest.raises(TypeError, match=removed_field):
        QuantileHeadConfig(device="cpu", **{removed_field: 0})


def test_quantile_head_processor_round_trip_preserves_state_prompt(tmp_path):
    step = QuantileHeadPrepareStateTokenizerProcessorStep(max_state_dim=2)
    pipeline = PolicyProcessorPipeline(steps=[step], name="quantile_head_preprocessor")
    pipeline.save_pretrained(tmp_path, config_filename="processor.json")

    saved = json.loads((tmp_path / "processor.json").read_text())
    assert saved["steps"][0]["registry_name"] == "quantile_head_prepare_state_tokenizer_processor_step"
    restored = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename="processor.json", local_files_only=True
    )
    assert isinstance(restored.steps[0], QuantileHeadPrepareStateTokenizerProcessorStep)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[0.0, 1.0]])},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick_up\nobject"]},
    }
    prepared = restored.steps[0](transition)
    assert prepared[TransitionKey.COMPLEMENTARY_DATA]["task"] == [
        "Task: pick up object, State: 128 255;\nAction: "
    ]


def test_pi05_processor_name_is_not_registered():
    with pytest.raises(KeyError, match="pi05_prepare_state_tokenizer_processor_step"):
        ProcessorStepRegistry.get("pi05_prepare_state_tokenizer_processor_step")

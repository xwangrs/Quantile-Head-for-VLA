#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("quantile_head")
@dataclass
class QuantileHeadConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)
    action_generation_mode: str = "regression"  # Options: "regression", "flow_matching"
    # Opt-in point regression baseline. False preserves the existing marginal quantile head.
    use_pure_regression_head: bool = False
    pure_regression_loss: str = "l1"  # Options: "l1", "l2"; used only by the pure regression head
    # Training/architecture ablations; saved with the checkpoint, not an evaluation decoder override.
    marginal_quantile_variant: str = "standard"
    # Options: "standard", "median_only", "stop_median_gradient", "independent".
    action_loss_mask_ratio: float = 0.1  # Random fraction of flattened action losses ignored per pass
    marginal_quantile_num_knots: int = 21  # Must be odd so the middle knot is the median
    # Learned knots span [min_level, 1-min_level]; diagnostic interpolation extrapolates the two tails.
    marginal_quantile_min_level: float = 0.025
    # Gap parameterization scale; zero raw gaps map to this value, but it is not a minimum gap.
    marginal_quantile_initial_gap: float = 0.02
    # Regression inference. ``median`` remains deterministic and preserves the historical behavior.
    # ``temporal_sample`` samples one quantile rank per action dimension, shares it across the
    # predicted chunk, and evolves it between replans with an AR(1) process.
    marginal_quantile_inference_mode: str = "median"  # Options: "median", "temporal_sample"
    marginal_quantile_sampling_temperature: float = 0.35
    marginal_quantile_sampling_min_level: float = 0.1
    marginal_quantile_sampling_max_level: float = 0.9
    marginal_quantile_sampling_ar_coefficient: float = 0.9
    marginal_quantile_sampling_seed: int = 1000
    # Matching action dimensions are kept at their median during temporal sampling.
    marginal_quantile_sampling_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    # Freeze the base VLM while keeping the action expert, action-side modules, and soft prompt trainable.
    train_expert_only: bool = False
    detach_action_from_vlm: bool = False  # Stop action-loss gradients at the VLM/action boundary
    soft_prompt_len: int = 16  # Number of trainable action-side prompt tokens prepended to action tokens
    train_soft_prompt: bool = True  # Keep soft prompts trainable when soft_prompt_len > 0
    soft_prompt_init_std: float = 0.02

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    optimizer_soft_prompt_lr: float | None = 1e-4  # Set None to use optimizer_lr
    optimizer_soft_prompt_weight_decay: float = 0.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.soft_prompt_len < 0:
            raise ValueError(f"soft_prompt_len ({self.soft_prompt_len}) must be >= 0")

        if self.soft_prompt_init_std <= 0:
            raise ValueError(f"soft_prompt_init_std ({self.soft_prompt_init_std}) must be > 0")

        if self.action_generation_mode not in ["regression", "flow_matching"]:
            raise ValueError(
                "action_generation_mode must be one of ['regression', 'flow_matching'], "
                f"got {self.action_generation_mode}"
            )

        if not isinstance(self.use_pure_regression_head, bool):
            raise ValueError("use_pure_regression_head must be a boolean")
        if self.use_pure_regression_head and self.action_generation_mode != "regression":
            raise ValueError("use_pure_regression_head=True requires action_generation_mode='regression'")
        if self.pure_regression_loss not in ["l1", "l2"]:
            raise ValueError("pure_regression_loss must be one of ['l1', 'l2']")

        quantile_variants = ("standard", "median_only", "stop_median_gradient", "independent")
        if self.marginal_quantile_variant not in quantile_variants:
            raise ValueError(f"marginal_quantile_variant must be one of {quantile_variants}")
        if self.marginal_quantile_variant != "standard" and (
            self.action_generation_mode != "regression" or self.use_pure_regression_head
        ):
            raise ValueError("marginal_quantile_variant ablations require a regression quantile head")
        if self.marginal_quantile_variant in ("median_only", "independent") and (
            self.marginal_quantile_inference_mode != "median"
        ):
            raise ValueError(
                f"marginal_quantile_variant={self.marginal_quantile_variant!r} requires median inference: "
                "unsupervised or potentially crossing quantiles cannot be used for distribution sampling"
            )

        if not 0 <= self.action_loss_mask_ratio < 1:
            raise ValueError(f"action_loss_mask_ratio ({self.action_loss_mask_ratio}) must be in [0, 1)")

        if self.marginal_quantile_num_knots < 3 or self.marginal_quantile_num_knots % 2 == 0:
            raise ValueError(
                "marginal_quantile_num_knots must be an odd integer >= 3, "
                f"got {self.marginal_quantile_num_knots}"
            )

        if not 0 < self.marginal_quantile_min_level < 0.5:
            raise ValueError(
                f"marginal_quantile_min_level must be in (0, 0.5), got {self.marginal_quantile_min_level}"
            )

        if self.marginal_quantile_initial_gap <= 0:
            raise ValueError(
                f"marginal_quantile_initial_gap must be > 0, got {self.marginal_quantile_initial_gap}"
            )

        if self.marginal_quantile_inference_mode not in ["median", "temporal_sample"]:
            raise ValueError(
                "marginal_quantile_inference_mode must be one of ['median', 'temporal_sample'], "
                f"got {self.marginal_quantile_inference_mode}"
            )

        if self.marginal_quantile_sampling_temperature < 0:
            raise ValueError(
                "marginal_quantile_sampling_temperature must be >= 0, "
                f"got {self.marginal_quantile_sampling_temperature}"
            )

        if not 0 <= self.marginal_quantile_sampling_min_level < 0.5:
            raise ValueError(
                "marginal_quantile_sampling_min_level must be in [0, 0.5), "
                f"got {self.marginal_quantile_sampling_min_level}"
            )

        if not 0.5 < self.marginal_quantile_sampling_max_level <= 1:
            raise ValueError(
                "marginal_quantile_sampling_max_level must be in (0.5, 1], "
                f"got {self.marginal_quantile_sampling_max_level}"
            )

        if not 0 <= self.marginal_quantile_sampling_ar_coefficient <= 1:
            raise ValueError(
                "marginal_quantile_sampling_ar_coefficient must be in [0, 1], "
                f"got {self.marginal_quantile_sampling_ar_coefficient}"
            )

        if not isinstance(self.marginal_quantile_sampling_seed, int) or isinstance(
            self.marginal_quantile_sampling_seed, bool
        ):
            raise ValueError(
                "marginal_quantile_sampling_seed must be an integer, "
                f"got {self.marginal_quantile_sampling_seed}"
            )

        if self.marginal_quantile_sampling_seed < 0:
            raise ValueError(
                f"marginal_quantile_sampling_seed must be >= 0, got {self.marginal_quantile_sampling_seed}"
            )

        if not isinstance(self.marginal_quantile_sampling_exclude_joints, list) or not all(
            isinstance(joint_name, str) and joint_name
            for joint_name in self.marginal_quantile_sampling_exclude_joints
        ):
            raise ValueError("marginal_quantile_sampling_exclude_joints must contain non-empty strings")

        if self.optimizer_soft_prompt_lr is not None and self.optimizer_soft_prompt_lr <= 0:
            raise ValueError(
                f"optimizer_soft_prompt_lr ({self.optimizer_soft_prompt_lr}) must be > 0 when set"
            )

        if self.optimizer_soft_prompt_weight_decay < 0:
            raise ValueError(
                f"optimizer_soft_prompt_weight_decay ({self.optimizer_soft_prompt_weight_decay}) must be >= 0"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

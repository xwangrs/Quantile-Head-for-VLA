from types import SimpleNamespace

from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig


def make_quantile_head_config_stub(**overrides):
    """Give tiny models the current schema while allowing deliberately invalid test inputs."""
    config = QuantileHeadConfig(device="cpu", soft_prompt_len=0)
    return SimpleNamespace(**(vars(config) | {"type": config.type} | overrides))

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Exercise experiment commands and orchestration without a model or simulator."""

import csv
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPT = REPO_ROOT / "sh/eval_quantile_ablations.sh"
DM_SCRIPT = REPO_ROOT / "sh/eval_quantile_dm.sh"
GIT_BASH = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
BASH = str(GIT_BASH) if os.name == "nt" and GIT_BASH.is_file() else shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required for quantile launcher tests")


def _run_script(script, output_root, *, seeds=("0",), overrides=None, fake_cli=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith("QUANTILE_")}
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    env.update({"QUANTILE_DRY_RUN": "1", "LC_ALL": "C.UTF-8"})
    env.update(overrides or {})
    arguments = [script.as_posix(), "checkpoint with spaces", output_root.as_posix(), "50", *seeds]
    if fake_cli is None:
        command = [BASH, "--noprofile", "--norc", *arguments]
    else:
        # An exported mock function shadows the actual CLI in every child Bash.
        # No Python policy/environment imports or real evaluation can occur.
        mock_launcher = fake_cli + '\nexport -f lerobot-eval\nexec bash "$@"'
        command = [BASH, "--noprofile", "--norc", "-c", mock_launcher, "quantile-test", *arguments]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def _dry_commands(result):
    assert result.returncode == 0, result.stdout + result.stderr
    commands = []
    for line in result.stdout.splitlines():
        if line.startswith("lerobot-eval "):
            arguments = shlex.split(line)
            commands.append(dict(argument.split("=", 1) for argument in arguments[1:]))
    return commands


def _variant(command):
    return Path(command["--output_dir"]).name


def _sampling_value(command, suffix):
    return float(command[f"--policy.marginal_quantile_sampling_{suffix}"])


def test_dm_dry_run_repeats_stochastic_variants_and_calibrates_once(tmp_path):
    output_root = tmp_path / "dry preview"
    commands = _dry_commands(_run_script(DM_SCRIPT, output_root, seeds=("3", "7")))
    calibrations = [
        command for command in commands if command["--quantile_head_quantile_decoder"] == "calibrate_width"
    ]
    evaluations = [
        command for command in commands if command["--quantile_head_quantile_decoder"] != "calibrate_width"
    ]

    assert len(calibrations) == 1
    assert len(evaluations) == 21
    expected = ["D", "E", "F", "F0", "G", "H", "I", "J", "K", "L", "M"]
    assert [_variant(command) for command in evaluations] == expected + [v for v in expected if v != "L"]
    assert [_variant(command) for command in evaluations].count("L") == 1
    assert evaluations[9]["--policy.marginal_quantile_sampling_seed"] == "3"
    assert calibrations[0]["--env.init_states"] == "false"
    assert calibrations[0]["--seed"] == "2000"
    assert calibrations[0]["--eval.n_episodes"] == "5"
    assert calibrations[0]["--policy.marginal_quantile_inference_mode"] == "median"
    assert all(command["--env.init_states"] == "true" for command in evaluations)
    assert all(command["--seed"] == "1000" for command in evaluations)
    assert all(command["--policy.path"] == "checkpoint with spaces" for command in commands)
    width_path = calibrations[0]["--quantile_head_quantile_width_output_path"]
    assert all(
        command["--quantile_head_quantile_fixed_width_path"] == width_path
        for command in evaluations
        if _variant(command) == "M"
    )
    assert not output_root.exists()


def test_dm_variant_settings_preserve_controlled_comparisons(tmp_path):
    commands = _dry_commands(_run_script(DM_SCRIPT, tmp_path / "dry preview"))
    variants = {
        _variant(command): command for command in commands if _variant(command) != "width_calibration"
    }
    expected = {
        "D": ("policy", 1.0, 0.2, 0.8, 0.35),
        "E": ("independent_rank", 0.0, 0.2, 0.8, 0.35),
        "F": ("gaussian_temporal", 0.7, 0.2, 0.8, 0.35),
        "F0": ("gaussian_temporal", 0.0, 0.2, 0.8, 0.35),
        "G": ("policy", 0.0, 0.2, 0.8, 0.35),
        "H": ("policy", 0.0, 0.3, 0.7, 0.35),
        "I": ("policy", 0.0, 0.4, 0.6, 0.35),
        "J": ("policy", 0.0, 0.2, 0.8, 0.15),
        "K": ("policy", 0.0, 0.2, 0.8, 1.0),
        "L": ("quantile_expectation", 0.0, 0.2, 0.8, 0.35),
        "M": ("fixed_width", 0.0, 0.2, 0.8, 0.35),
    }
    for name, (decoder, rho, lo, hi, temperature) in expected.items():
        command = variants[name]
        assert command["--quantile_head_quantile_decoder"] == decoder
        assert _sampling_value(command, "ar_coefficient") == rho
        assert _sampling_value(command, "min_level") == lo
        assert _sampling_value(command, "max_level") == hi
        assert _sampling_value(command, "temperature") == temperature
        assert command["--policy.n_action_steps"] == "10"
        assert command["--policy.marginal_quantile_sampling_exclude_joints"] == '["gripper"]'
    assert variants["L"]["--quantile_head_quantile_expectation_nodes"] == "128"


def test_reference_overrides_leave_explicit_range_and_temperature_ablations_intact(tmp_path):
    commands = _dry_commands(
        _run_script(
            DM_SCRIPT,
            tmp_path / "dry preview",
            overrides={
                "QUANTILE_EVAL_VARIANTS": "B G H I J K L",
                "QUANTILE_EVAL_MIN_LEVEL": "0.25",
                "QUANTILE_EVAL_MAX_LEVEL": "0.75",
                "QUANTILE_EVAL_TEMPERATURE": "0.6",
                "QUANTILE_EXPECTATION_NODES": "64",
            },
        )
    )
    variants = {_variant(command): command for command in commands}
    for name in ("B", "J", "K", "L"):
        assert _sampling_value(variants[name], "min_level") == 0.25
        assert _sampling_value(variants[name], "max_level") == 0.75
    for name in ("B", "G", "H", "I", "L"):
        assert _sampling_value(variants[name], "temperature") == 0.6
    assert _sampling_value(variants["G"], "min_level") == 0.2
    assert _sampling_value(variants["H"], "min_level") == 0.3
    assert _sampling_value(variants["I"], "min_level") == 0.4
    assert _sampling_value(variants["J"], "temperature") == 0.15
    assert _sampling_value(variants["K"], "temperature") == 1.0
    assert variants["L"]["--quantile_head_quantile_expectation_nodes"] == "64"


def test_external_width_dry_run_does_not_require_file_or_calibration(tmp_path):
    width_path = tmp_path / "not yet available.json"
    commands = _dry_commands(
        _run_script(
            DM_SCRIPT,
            tmp_path / "dry preview",
            seeds=("0", "1"),
            overrides={"QUANTILE_EVAL_VARIANTS": "M", "QUANTILE_FIXED_WIDTH_PATH": width_path.as_posix()},
        )
    )
    assert len(commands) == 2
    assert all(command["--quantile_head_quantile_decoder"] == "fixed_width" for command in commands)
    assert all(
        command["--quantile_head_quantile_fixed_width_path"] == width_path.as_posix() for command in commands
    )
    assert not width_path.exists()


def test_core_default_abc_and_deterministic_deduplication_are_preserved(tmp_path):
    commands = _dry_commands(_run_script(CORE_SCRIPT, tmp_path / "dry preview", seeds=("0", "1")))
    assert [_variant(command) for command in commands] == ["A", "B", "C", "B", "C"]
    assert commands[0]["--policy.marginal_quantile_inference_mode"] == "median"
    for command in commands:
        assert _sampling_value(command, "min_level") == 0.1
        assert _sampling_value(command, "max_level") == 0.9
        assert _sampling_value(command, "temperature") == 0.35


@pytest.mark.parametrize(
    "overrides",
    [
        {"QUANTILE_EVAL_VARIANTS": ""},
        {"QUANTILE_EVAL_VARIANTS": "D D"},
        {"QUANTILE_EVAL_VARIANTS": "D UNKNOWN"},
        {"QUANTILE_EVAL_VARIANTS": "D\nE"},
        {"QUANTILE_EVAL_MIN_LEVEL": "0.5"},
        {"QUANTILE_EVAL_MAX_LEVEL": "0.4"},
        {"QUANTILE_EVAL_TEMPERATURE": "-0.1"},
        {"QUANTILE_EVAL_TEMPERATURE": "nan"},
        {"QUANTILE_EVAL_TEMPERATURE": "1e999"},
        {"QUANTILE_EXPECTATION_NODES": "1"},
        {"QUANTILE_CALIBRATION_EPISODES": "0"},
        {"QUANTILE_EVAL_SEED": "-1"},
        {"QUANTILE_DRY_RUN": "yes"},
    ],
)
def test_invalid_experiment_settings_fail_before_any_output(tmp_path, overrides):
    output_root = tmp_path / "invalid preview"
    result = _run_script(DM_SCRIPT, output_root, overrides=overrides)
    assert result.returncode == 2
    assert not output_root.exists()
    assert "lerobot-eval " not in result.stdout


def test_duplicate_sampler_seeds_are_rejected(tmp_path):
    result = _run_script(DM_SCRIPT, tmp_path / "dry preview", seeds=("0", "0"))
    assert result.returncode == 2
    assert "Duplicate SAMPLER_SEED" in result.stderr


def test_zero_temperature_and_full_probability_range_are_valid(tmp_path):
    commands = _dry_commands(
        _run_script(
            DM_SCRIPT,
            tmp_path / "dry preview",
            overrides={
                "QUANTILE_EVAL_VARIANTS": "B",
                "QUANTILE_EVAL_MIN_LEVEL": "0",
                "QUANTILE_EVAL_MAX_LEVEL": "1",
                "QUANTILE_EVAL_TEMPERATURE": "0",
            },
        )
    )
    assert _sampling_value(commands[0], "temperature") == 0


FAKE_CLI = r"""
lerobot-eval() {
    local run_dir='' decoder='' width_output='' width_input='' argument
    for argument in "$@"; do
        case "$argument" in
            --output_dir=*) run_dir="${argument#*=}" ;;
            --quantile_head_quantile_decoder=*) decoder="${argument#*=}" ;;
            --quantile_head_quantile_width_output_path=*) width_output="${argument#*=}" ;;
            --quantile_head_quantile_fixed_width_path=*) width_input="${argument#*=}" ;;
        esac
    done
    printf '%s\n' "$decoder" >> "$QUANTILE_TEST_INVOCATIONS"
    if [[ $decoder == calibrate_width ]]; then
        if [[ ${QUANTILE_TEST_CALIBRATION_FAIL:-0} == 1 ]]; then
            return 7
        fi
        printf '{"widths": [[1.0]]}\n' > "$width_output"
    elif [[ $decoder == fixed_width && ! -f $width_input ]]; then
        return 8
    fi
    printf '{}\n' > "$run_dir/eval_info.json"
}
"""


def _read_status(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def test_calibration_failure_skips_only_m_and_preserves_logs(tmp_path):
    output_root = tmp_path / "failed calibration"
    invocations = tmp_path / "invocations.txt"
    overrides = {
        "QUANTILE_DRY_RUN": "0",
        "QUANTILE_EVAL_VARIANTS": "B M E",
        "QUANTILE_TEST_INVOCATIONS": invocations.as_posix(),
        "QUANTILE_TEST_CALIBRATION_FAIL": "1",
    }
    result = _run_script(DM_SCRIPT, output_root, overrides=overrides, fake_cli=FAKE_CLI)

    assert result.returncode == 1, result.stdout + result.stderr
    assert invocations.read_text().splitlines() == ["calibrate_width", "policy", "independent_rank"]
    status = _read_status(output_root / "status.tsv")
    assert [(row["variant"], row["exit_code"]) for row in status] == [("B", "0"), ("M", "1"), ("E", "0")]
    assert "M skipped because width calibration failed" in (output_root / "seed_0/M/eval.log").read_text()
    calibration_status = _read_status(output_root / "width_calibration/calibration_status.tsv")
    assert calibration_status[0]["exit_code"] == "7"
    log_before_retry = (output_root / "run_all.log").read_bytes()

    retry = _run_script(DM_SCRIPT, output_root, overrides=overrides, fake_cli=FAKE_CLI)
    assert retry.returncode == 1
    assert "Use a new OUTPUT_ROOT" in retry.stderr
    assert (output_root / "run_all.log").read_bytes() == log_before_retry
    assert invocations.read_text().splitlines() == ["calibrate_width", "policy", "independent_rank"]


def test_successful_calibration_is_reused_and_deterministic_variants_run_once(tmp_path):
    output_root = tmp_path / "successful calibration"
    invocations = tmp_path / "invocations.txt"
    result = _run_script(
        DM_SCRIPT,
        output_root,
        seeds=("2", "9"),
        overrides={
            "QUANTILE_DRY_RUN": "0",
            "QUANTILE_EVAL_VARIANTS": "A L M",
            "QUANTILE_TEST_INVOCATIONS": invocations.as_posix(),
        },
        fake_cli=FAKE_CLI,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert invocations.read_text().splitlines() == [
        "calibrate_width",
        "policy",
        "quantile_expectation",
        "fixed_width",
        "fixed_width",
    ]
    status = _read_status(output_root / "status.tsv")
    assert [(row["variant"], row["sampler_seed"]) for row in status] == [
        ("A", "2"),
        ("L", "2"),
        ("M", "2"),
        ("M", "9"),
    ]
    assert all(row["exit_code"] == "0" for row in status)
    assert (output_root / "width_calibration/fixed_width.json").is_file()
    assert not (output_root / "seed_9/A").exists()
    assert not (output_root / "seed_9/L").exists()

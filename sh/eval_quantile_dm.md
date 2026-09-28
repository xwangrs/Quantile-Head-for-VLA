# Quantile D-M inference experiments

Use one existing quantile checkpoint. These scripts do not start training. Run from
the repository in the Linux environment that already supports `lerobot-eval` and
LIBERO. Experiments run sequentially on the current device.

```bash
# Preview every command, including M's separate width calibration.
QUANTILE_DRY_RUN=1 bash sh/eval_quantile_dm.sh \
  /path/to/pretrained_model outputs/quantile_dm_preview 50 0

# Run D-M at one sampler seed (50 episodes per task, 500 for libero_10).
bash sh/eval_quantile_dm.sh \
  /path/to/pretrained_model outputs/quantile_dm_seed0 50 0

# Repeat stochastic variants at three sampler seeds; L runs only once.
bash sh/eval_quantile_dm.sh \
  /path/to/pretrained_model outputs/quantile_dm_three_seeds 50 0 1 2
```

The wrapper defaults to `D E F F0 G H I J K L M`, reference range `[0.2,0.8]`, and
temperature `0.35`. The original `eval_quantile_ablations.sh` keeps its defaults:
`A B C`, range `[0.1,0.9]`, temperature `0.35`. Both accept the same positional
arguments and environment variables. With no positional arguments, both use the
checkpoint `/root/autodl-fs/outputs/libero_quantile_action_prompt/checkpoints/last/pretrained_model`,
a timestamped output directory under `/root/autodl-fs`, 50 episodes per task, and seed 0.

All variants execute 10 actions before replanning. The gripper uses the median
except in `C_all`. Shared ranks mean each action dimension keeps its own rank
across the predicted chunk; ranks are independent between dimensions. Evaluation
uses fixed LIBERO initial states and `QUANTILE_EVAL_SEED=1000` by default. The
positional seeds change the quantile sampler, not the evaluation environment seed.

| Variant | Range | Temperature | Difference and comparison |
| --- | --- | --- | --- |
| A | Unused | Unused | Median only; deterministic reference, first sampler seed only. |
| B | Reference | Reference | Chunk-shared ranks; resample independently at each planning call, rho=0. |
| C | Reference | Reference | Like B with cross-chunk AR correlation rho=0.7. |
| C_all | Reference | Reference | Like C, also sampling the gripper. |
| D | Reference | Reference | rho=1: one random rank per dimension for the whole episode. Actions still change with observations. |
| E | Reference | Reference | Independent ranks for each time step and dimension, rho=0. Compare with B. |
| F | Reference | Reference | Gaussian curve fitted from median and Q0.9-Q0.1 width, rho=0.7. Compare with C. |
| F0 | Reference | Reference | Same Gaussian decoder with rho=0. Compare with B. |
| G | [0.2,0.8] | Reference | Like B, fixed range override. |
| H | [0.3,0.7] | Reference | Like B, narrower range. |
| I | [0.4,0.6] | Reference | Like B, very narrow range. |
| J | Reference | 0.15 | Like B, lower temperature. |
| K | Reference | 1.0 | Like B, uniform ranks inside the selected range. |
| L | Reference | Reference | Deterministic expectation under B's rank distribution, evaluated by fixed numerical quadrature; first sampler seed only. Compare A/L/B. |
| M | Reference | Reference | Like B with calibrated fixed Q0.9-Q0.1 widths per horizon step and dimension, retaining the current median and normalized quantile curve. Compare with B. |

`G` duplicates `B` under the wrapper's default range. Include B explicitly when
you want the controls recorded alongside these experiments:

```bash
QUANTILE_EVAL_VARIANTS="A B C D E F F0 G H I J K L M" \
bash sh/eval_quantile_dm.sh \
  /path/to/pretrained_model outputs/quantile_all_seed0 50 0

# First run just the requested narrow-range scan.
QUANTILE_EVAL_VARIANTS="A G H I" \
bash sh/eval_quantile_dm.sh \
  /path/to/pretrained_model outputs/quantile_ranges 50 0 1 2

# Then compare mechanisms at a selected common range and temperature.
QUANTILE_EVAL_VARIANTS="A B C E F F0 L M" \
QUANTILE_EVAL_MIN_LEVEL=0.3 QUANTILE_EVAL_MAX_LEVEL=0.7 \
QUANTILE_EVAL_TEMPERATURE=0.35 \
bash sh/eval_quantile_dm.sh \
  /path/to/pretrained_model outputs/quantile_mechanisms 50 0 1 2
```

G/H/I always override the range; J/K always override the temperature. All other
settings come from the reference variables. Keep range and temperature identical
within B/E, C/F, B/F0, and B/L/M comparisons. Old scores at `[0.1,0.9]` do not provide
a controlled comparison for the wrapper's new `[0.2,0.8]` defaults.

## L: deterministic expectation

`QUANTILE_EXPECTATION_NODES=128` sets the number of fixed integration nodes (at
least 2). L approximates the expectation of the action under the selected sampler
distribution, rather than averaging the stored quantiles uniformly. It ignores
the sampler seed. A difference between A and L can arise from an asymmetric
quantile curve even when the rank distribution is symmetric around 0.5.

## M: calibration is separate from evaluation

If `QUANTILE_FIXED_WIDTH_PATH` is unset, the script first runs a median policy on
random LIBERO initial states (`--env.init_states=false`) with 5 episodes per task
and environment seed 2000. It records the RMS of the predicted Q0.9-Q0.1 widths
for each horizon step and action dimension, then uses those fixed widths for M.
This calibration is run once, shared by all M sampler seeds, and stored separately
under `OUTPUT_ROOT/width_calibration/`. Its scores are not experiment results.
Random initialization separates the calibration procedure from the fixed test
initial states; it does not guarantee their state distributions are disjoint.

```bash
# Control the size and seed of the separate calibration.
QUANTILE_EVAL_VARIANTS="B M" \
QUANTILE_CALIBRATION_EPISODES=10 QUANTILE_CALIBRATION_SEED=2000 \
bash sh/eval_quantile_dm.sh \
  /path/to/pretrained_model outputs/quantile_fixed_width 50 0 1 2

# Reuse a width file from a separate calibration with the same checkpoint.
QUANTILE_EVAL_VARIANTS="B M" \
QUANTILE_FIXED_WIDTH_PATH=outputs/quantile_fixed_width/width_calibration/fixed_width.json \
bash sh/eval_quantile_dm.sh \
  /path/to/pretrained_model outputs/quantile_fixed_width_repeat 50 3 4
```

F/F0 and M retain information from the trained quantile model; they are decoding
ablations, not separately trained baselines. Neither the Gaussian width fit nor
M's RMS width calibration guarantees identical realized action-noise variance.
Inspect action deviations as well as success rates before attributing a score
difference solely to distribution shape or state-dependent width.

## Outputs and failures

Each run saves its exact command, `eval.log`, and `eval_info.json` in
`OUTPUT_ROOT/seed_<seed>/<variant>/`. The root `status.tsv` records per-run exit
codes and paths; `run_all.log` records progress. Set `QUANTILE_EVAL_VIZ=1` to enable
the existing action diagnostic output. Existing run logs and run directories are
never overwritten: use a fresh output root for each invocation.

Dry run prints commands without creating directories or requiring a width file.
If automatic calibration fails, M is recorded as failed and skipped while the
other variants continue. Other evaluation failures also allow later variants to
run, and the script exits nonzero if any experiment fails. Calibration has its
own `calibration_status.tsv` and does not enter the main evaluation status table.

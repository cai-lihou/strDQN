# StrDQN

StrDQN is a reinforcement-learning framework for temporal influence maximization under the Strict TW-IC diffusion model. The code formulates seed selection and activation timing as a sequential decision-making problem, where the agent can either select a seed node or use a `WAIT` action to delay activation until a more useful temporal window.

This repository contains the core StrDQN implementation, data preprocessing, TGAT pretraining, ablation experiments, fair re-evaluation, result aggregation, and plotting utilities.

## Repository Structure

```text
.
|-- run_experiments.py              # Main experiment launcher
|-- scripts/
|   |-- preprocess_data.py          # Raw temporal edge preprocessing
|   |-- pretrain_tgat.py            # TGAT pretraining
|   |-- train_strdqn.py             # Full StrDQN training/evaluation
|   |-- train_ablation.py           # Ablation training/evaluation
|   |-- fair_evaluate.py            # Fair strict TW-IC re-evaluation
|   |-- aggregate_results.py        # Mean/std/statistical aggregation
|   |-- plot_results.py             # Main result plotting
|   |-- diagnose_node_schedule.py   # Node schedule diagnostics
|   |-- plot_activity.py            # Temporal activity plotting
|   |-- plot_seed_activity_alignment.py
|   `-- plot_wait_timing.py
|-- src/
|   |-- common/
|   |   |-- result_io.py            # Result path, Excel, and aggregation helpers
|   |   `-- utils.py
|   |-- diffusion/
|   |   `-- strict_tw_ic.py         # Strict TW-IC model
|   `-- models/
|       |-- graph.py                # Temporal neighbor finder
|       `-- module.py               # TGAT modules
|-- processed/                      # Local data directory, not tracked
|-- saved_models/                   # Local model checkpoint directory, not tracked
`-- result_new_*/                   # Local result directories, not tracked
```

## Environment

Python 3.8 or 3.9 is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version if GPU acceleration is required. See the official PyTorch installation instructions for the correct CUDA build.

## Data Preparation

Place raw temporal edge files under `processed/`. A raw file should contain at least three columns corresponding to two nodes and one timestamp. Both common formats are supported:

```text
u,v,t
```

or

```text
t,u,v
```

The preprocessing script infers the timestamp column, remaps node IDs to a contiguous range, generates node/edge features, and writes the processed pickle used by the experiments.

Example:

```bash
python scripts/preprocess_data.py --dataset thiers_2012
```

Expected output:

```text
processed/thiers_2012_main.pkl
```

## TGAT Pretraining

Pretrain TGAT before running StrDQN:

```bash
python scripts/pretrain_tgat.py --data thiers_2012
```

Expected checkpoint path:

```text
saved_models/-attn-prod-thiers_2012.pth
```

## Running Experiments

Run a small smoke test:

```bash
python run_experiments.py \
  --dataset thiers_2012 \
  --parts main \
  --seeds 0 \
  --budgets 10 \
  --durations 0.001 \
  --episodes 1 \
  --gpu 0 \
  --result-dir ./result_new_thiers_2012_test \
  --no-resume
```

Run the core workflow:

```bash
python run_experiments.py \
  --dataset thiers_2012 \
  --parts main,ablation,sensitivity,fair_eval,aggregate,plot \
  --seeds 0,1,2,3,4 \
  --budgets 10,20,30,50 \
  --durations 0.001,0.005,0.01 \
  --ablation-budgets 50 \
  --ablation-durations 0.005 \
  --sensitivity-budget 30 \
  --sensitivity-duration 0.005 \
  --episodes 1200 \
  --gpu 0 \
  --result-dir ./result_new_thiers_2012
```

Use multiple GPUs by passing visible GPU IDs:

```bash
python run_experiments.py \
  --dataset thiers_2012 \
  --parts main,ablation,sensitivity,fair_eval,aggregate,plot \
  --seeds 0,1,2,3,4 \
  --budgets 10,20,30,50 \
  --durations 0.001,0.005,0.01 \
  --episodes 1200 \
  --gpus 0,1 \
  --parallel-jobs 2 \
  --result-dir ./result_new_thiers_2012
```

## Experiment Parts

`run_experiments.py` supports the following parts:

```text
main        Full StrDQN
ablation    Ablation variants
sensitivity Parameter sensitivity analysis
fair_eval   Re-evaluate saved seed schedules with strict TW-IC
aggregate   Aggregate mean/std and statistical tests
plot        Plot aggregated results
```

The supported ablation modes are:

```text
no_wait
no_action_bias
unified
no_wait_compensation
```

## Output Files

Results are written to the selected result directory, for example:

```text
result_new_thiers_2012/
|-- result_thiers_2012_minimal.xlsx
|-- result_thiers_2012_no_wait_minimal.xlsx
|-- thiers_2012_fair_eval_minimal.xlsx
|-- thiers_2012_minimal_main_results.xlsx
|-- thiers_2012_minimal_ablation_results.xlsx
|-- thiers_2012_minimal_sensitivity_results.xlsx
`-- minimal_plots/
```

Generated result files, plots, model checkpoints, and processed binary files are intentionally ignored by Git.

## Notes

- Raw datasets, processed `.pkl` files, result Excel files, figures, and trained checkpoints are not tracked.
- If you run scripts from inside `scripts/`, the repository root is automatically added to `sys.path` so imports from `src/` work correctly.

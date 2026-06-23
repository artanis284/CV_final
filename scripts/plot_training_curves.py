import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ACT training curves exported from WandB/SwanLab CSV.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--multi-env", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def find_column(df, candidates):
    normalized = {c.lower().replace("/", "_").replace("-", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace("/", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Cannot find any column in {candidates}; available={list(df.columns)}")


def plot_metric(base, multi, metric_candidates, ylabel, out_path):
    step_b = find_column(base, ["step", "global_step", "_step"])
    step_m = find_column(multi, ["step", "global_step", "_step"])
    metric_b = find_column(base, metric_candidates)
    metric_m = find_column(multi, metric_candidates)

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(base[step_b], base[metric_b], label="Env-B only", linewidth=2)
    plt.plot(multi[step_m], multi[metric_m], label="Env-A+B+C", linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(args.baseline)
    multi = pd.read_csv(args.multi_env)

    plot_metric(
        baseline,
        multi,
        ["train_action_l1_loss", "action_l1_loss", "train/loss/action_l1", "loss"],
        "Action L1 Loss",
        args.out_dir / "act_action_l1_loss.png",
    )
    plot_metric(
        baseline,
        multi,
        ["val_success_rate", "success_rate", "eval_success_rate", "eval/success_rate"],
        "Validation Success Rate",
        args.out_dir / "act_success_rate.png",
    )


if __name__ == "__main__":
    main()

import argparse
import subprocess
from pathlib import Path

import yaml


def run(cmd, cwd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Thin wrapper for LeRobot ACT train/eval.")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train")
    train.add_argument("--repo", required=True, type=Path)
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--dry-run", action="store_true")

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--repo", required=True, type=Path)
    eval_p.add_argument("--checkpoint", required=True, type=Path)
    eval_p.add_argument("--env", required=True, choices=["A", "B", "C", "D"])
    eval_p.add_argument("--output", required=True, type=Path)
    eval_p.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = args.repo.resolve()
    if args.mode == "train":
        if not args.dry_run and not repo.exists():
            raise FileNotFoundError(f"LeRobot repo not found: {repo}")
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        args.output.mkdir(parents=True, exist_ok=True)
        envs = ",".join(cfg["dataset"]["environments"])
        train_cfg = cfg["training"]
        model_cfg = cfg["model"]
        log_cfg = cfg["logging"]

        cmd = [
            "python", "-m", "lerobot.scripts.train",
            f"dataset.repo_id=local/calvin_{envs.lower()}",
            f"dataset.root={Path(cfg['dataset']['root']).resolve()}",
            "policy.type=act",
            f"policy.chunk_size={model_cfg['chunk_size']}",
            f"policy.kl_weight={model_cfg['kl_weight']}",
            f"batch_size={train_cfg['batch_size']}",
            f"steps={train_cfg['epochs']}",
            f"optimizer.lr={train_cfg['learning_rate']}",
            f"wandb.enable={str(log_cfg['backend'] == 'wandb').lower()}",
            f"wandb.project={log_cfg['project']}",
            f"wandb.name={log_cfg['run_name']}",
            f"output_dir={args.output.resolve()}",
        ]
        print("+", " ".join(cmd))
        if not args.dry_run:
            run(cmd, cwd=repo)

    if args.mode == "eval":
        if not args.dry_run and not repo.exists():
            raise FileNotFoundError(f"LeRobot repo not found: {repo}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python", "-m", "lerobot.scripts.eval",
            f"policy.path={args.checkpoint.resolve()}",
            f"env=calvin_{args.env}",
            f"output={args.output.resolve()}",
        ]
        print("+", " ".join(cmd))
        if not args.dry_run:
            run(cmd, cwd=repo)


if __name__ == "__main__":
    main()

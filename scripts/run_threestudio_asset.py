import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Launch threestudio SDS text-to-3D asset generation.")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repo = args.repo.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not repo.exists():
        raise FileNotFoundError(f"threestudio repo not found: {repo}")

    prompt = cfg["prompt"]
    steps = cfg.get("training", {}).get("max_steps", 10000)
    lr = cfg.get("training", {}).get("learning_rate", 0.01)

    cmd = [
        "python", "launch.py",
        "--config", "configs/dreamfusion-sd.yaml",
        "--train",
        "--gpu", "0",
        f"system.prompt_processor.prompt={prompt}",
        f"trainer.max_steps={steps}",
        f"system.geometry.isosurface_threshold=25.0",
        f"system.optimizer.args.lr={lr}",
        f"exp_root_dir={out}",
    ]
    print("+", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=repo, check=True)


if __name__ == "__main__":
    main()

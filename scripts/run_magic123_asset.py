import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Magic123 single-image-to-3D optimization.")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repo = args.repo.resolve()
    image = args.image.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not repo.exists():
        raise FileNotFoundError(f"Magic123 repo not found: {repo}")
    if not args.dry_run and not image.exists():
        raise FileNotFoundError(f"Input image not found: {image}")

    coarse_steps = cfg.get("training", {}).get("coarse_steps", 5000)
    refine_steps = cfg.get("training", {}).get("refine_steps", 5000)
    cmd = [
        "python", "run.py",
        "--image", str(image),
        "--workspace", str(out),
        "--coarse_iters", str(coarse_steps),
        "--refine_iters", str(refine_steps),
    ]
    print("+", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=repo, check=True)


if __name__ == "__main__":
    main()

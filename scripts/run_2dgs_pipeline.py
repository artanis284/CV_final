import argparse
import subprocess
from pathlib import Path

import yaml


def run(cmd, cwd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run COLMAP preprocessing, 2DGS training, rendering, and metrics.")
    parser.add_argument("--repo", required=True, type=Path, help="Path to 2D Gaussian Splatting repo.")
    parser.add_argument("--source", required=True, type=Path, help="Dataset/source directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output model directory.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-colmap", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repo = args.repo.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not repo.exists():
        raise FileNotFoundError(f"2DGS repo not found: {repo}")
    if not source.exists():
        raise FileNotFoundError(f"Source data not found: {source}")

    resolution = str(cfg.get("resolution", 2))
    iterations = str(cfg.get("iterations", 30000))

    if not args.skip_colmap:
        convert = repo / "convert.py"
        if convert.exists():
            run(["python", str(convert), "-s", str(source)], cwd=repo)
        else:
            print("convert.py not found; assuming COLMAP data already exists.")

    train = repo / "train.py"
    train_cmd = [
        "python", str(train),
        "-s", str(source),
        "-m", str(output),
        "-r", resolution,
        "--iterations", iterations,
    ]
    if cfg.get("eval", True):
        train_cmd.append("--eval")
    run(train_cmd, cwd=repo)

    if not args.skip_render:
        render = repo / "render.py"
        metrics = repo / "metrics.py"
        if render.exists():
            run(["python", str(render), "-m", str(output)], cwd=repo)
        if metrics.exists() and cfg.get("eval", True):
            run(["python", str(metrics), "-m", str(output)], cwd=repo)


if __name__ == "__main__":
    main()

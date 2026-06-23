import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Collect final metrics into CSV/LaTeX tables.")
    parser.add_argument("--metrics-json", nargs="+", type=Path, required=True)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-tex", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in args.metrics_json:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source_file"] = str(path)
        rows.append(data)
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    args.out_tex.write_text(df.to_latex(index=False, float_format="%.4f"), encoding="utf-8")


if __name__ == "__main__":
    main()

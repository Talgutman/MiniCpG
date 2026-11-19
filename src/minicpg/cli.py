"""Command-line prediction with the locked MiniCpG tissue model."""
from __future__ import annotations

import argparse
from pathlib import Path
from minicpg.inference import predict_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Sample-by-probe TSV or CSV.")
    parser.add_argument("--output", required=True, help="Prediction and coverage-status TSV.")
    parser.add_argument("--model", default="model/minicpg_k30_model_bundle.pkl.gz", help="Trusted local model pickle.")
    parser.add_argument("--include-probabilities", action="store_true")
    args = parser.parse_args()
    if Path(args.output).resolve() in {Path(args.input).resolve(), Path(args.model).resolve()}:
        parser.error("Output must differ from input and model paths")
    try:
        result = predict_file(args.input, args.model, include_probabilities=args.include_probabilities)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    n_predicted = int(result["status"].eq("predicted").sum())
    print(f"Wrote {len(result)} rows: {n_predicted} predicted, "
          f"{len(result) - n_predicted} below coverage threshold")
    return 0

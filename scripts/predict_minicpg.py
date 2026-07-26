#!/usr/bin/env python3
"""Run inference with the locked MiniCpG K=30 model."""

from __future__ import annotations

import argparse
from pathlib import Path

from minicpg.inference import predict_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Input TSV/CSV with one row per sample and MiniCpG CpG columns.')
    parser.add_argument('--output', required=True, help='Output prediction TSV path.')
    parser.add_argument('--model', default='model/minicpg_k30_model_bundle.pkl.gz', help='Compressed MiniCpG model bundle.')
    parser.add_argument('--include-probabilities', action='store_true', help='Append one probability column per modeled class.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = predict_file(args.input, args.model, include_probabilities=args.include_probabilities)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep='\t', index=False)
    print(f'Wrote {len(result)} predictions to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

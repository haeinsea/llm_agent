from __future__ import annotations

import argparse

from src.models.train_adaptable import main as train_adaptable_main
from src.models.train_invariant import main as train_invariant_main
from src.utils.experiment import get_seed_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["all", "adaptable", "invariant"],
        default="all",
        help="Select which SOTA baseline family to train.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit seed list. Defaults to configs/experiment.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(get_seed_list() if args.seeds is None else args.seeds)
    run_adaptable = args.only in {"all", "adaptable"}
    run_invariant = args.only in {"all", "invariant"}

    print(f"[START] train_sota_models seeds={seeds} only={args.only}", flush=True)
    for seed in seeds:
        if run_adaptable:
            train_adaptable_main(seed=seed)
        if run_invariant:
            train_invariant_main(seed=seed)
    print("[DONE] train_sota_models", flush=True)


if __name__ == "__main__":
    main()

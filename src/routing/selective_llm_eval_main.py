from __future__ import annotations

import argparse

from src.routing.selective_llm_eval import run_main_eval, run_main_eval_q_values_only, run_main_eval_selected_q_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-q-only",
        action="store_true",
        help="Recompute only the selected q on val/main, including the default ablation modes, without rerunning the full q-sweep.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        help="Optional subset of main modes to recompute when --selected-q-only is used. Example: no_llm ensemble_only selective_no_graph selective_no_filter",
    )
    parser.add_argument(
        "--q-values",
        nargs="+",
        type=float,
        help="Optional subset of q values to recompute for q-sweep rows only. This updates selective/no_llm main results without rerunning the full sweep.",
    )
    args = parser.parse_args()

    if args.selected_q_only and args.q_values:
        raise ValueError("--selected-q-only and --q-values cannot be used together.")

    if args.selected_q_only:
        _, summary = run_main_eval_selected_q_only(modes=args.modes)
    elif args.q_values:
        _, summary = run_main_eval_q_values_only(args.q_values)
    else:
        _, summary = run_main_eval()
    sub = summary[summary["dataset"].isin(["val", "main"])].copy()
    print("[selective_llm_eval_main] completed")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()

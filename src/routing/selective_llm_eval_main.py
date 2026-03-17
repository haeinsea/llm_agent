from __future__ import annotations

from src.routing.selective_llm_eval import run_main_eval


def main() -> None:
    _, summary = run_main_eval()
    sub = summary[summary["dataset"].isin(["val", "main"])].copy()
    print("[selective_llm_eval_main] completed")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()

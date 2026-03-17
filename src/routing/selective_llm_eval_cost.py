from __future__ import annotations

from src.routing.selective_llm_eval import run_cost_eval


def main() -> None:
    _, summary = run_cost_eval()
    sub = summary[summary["dataset"] == "cost"].copy()
    print("[selective_llm_eval_cost] completed")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()

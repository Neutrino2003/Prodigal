"""Evaluation metrics and reporting."""

import json
from pathlib import Path


def load_eval_results(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def print_eval_report(results: dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    conversations = results.get("conversations", [])
    aggregate = results.get("aggregate_scores", {})

    print(f"\nTotal Conversations: {len(conversations)}\n")

    print("Individual Scores:")
    print("-" * 60)
    for conv in conversations:
        persona = conv.get("persona", "Unknown")
        score = conv.get("score", 0)
        phase = conv.get("final_phase", "unknown")
        verified = conv.get("verified", False)
        print(f"{persona:<40} {score:6.1f}/100")
        print(f"  Phase: {phase}, Verified: {verified}")

    print("\n" + "-" * 60)

    if aggregate:
        avg = aggregate.get("average", 0)
        print(f"\nAggregate Score: {avg:.1f}/100")
        if conversations:
            scores = [c.get("score", 0) for c in conversations]
            print(f"Range: {min(scores):.1f} - {max(scores):.1f}")

    print("\n" + "=" * 60)


def get_latest_eval_results() -> dict | None:
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None
    json_files = sorted(results_dir.glob("eval_*.json"), reverse=True)
    if not json_files:
        return None
    return load_eval_results(str(json_files[0]))


if __name__ == "__main__":
    results = get_latest_eval_results()
    if results:
        print_eval_report(results)
    else:
        print("No evaluation results found.")

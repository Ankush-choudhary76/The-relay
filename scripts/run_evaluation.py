"""Script to execute retrieval evaluation suite across Setting A, B, and C."""

import asyncio
import json
from pathlib import Path
from relay.config import settings
from relay.evaluation import load_evaluation_questions, run_evaluation_suite
from relay.papers import extract_all_papers
from relay.standin_provider import StandinProvider


async def main():
    print("Preparing papers and evaluation suite...")
    extracted_papers = extract_all_papers()
    questions_file = settings.data_dir / "questions.jsonl"
    questions = load_evaluation_questions(questions_file)

    print(f"Loaded {len(questions)} evaluation questions.")
    provider = StandinProvider()

    results = await run_evaluation_suite(
        questions=questions,
        extracted_papers=extracted_papers,
        provider=provider,
        source_name="stand-in",
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION RESULTS (Table 3 Data):")
    print("=" * 70)
    for setting_key, res in results.items():
        print(f"\n{res['setting_name']}")
        print(f"  Questions Asked: {res['questions_asked']}")
        print(f"  Correct and Cited: {res['correct_and_cited_pct']}%")
        print(f"  Wrong: {res['wrong_pct']}%")
        print(f"  Correctly Refused: {res['refused_correctly_pct']}%")
        print(f"  Tokens per Answer: {res['tokens_per_answer']}")
        print(f"  TTFT p50 / p95: {res['ttft_p50']}ms / {res['ttft_p95']}ms")
        print(f"  Total Time p50 / p95: {res['total_p50']}ms / {res['total_p95']}ms")

    # Save summary json
    summary_path = settings.data_dir / "measurements" / "retrieval_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation summary to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())


#!/usr/bin/env python3
"""
AutismBench -CLI entry point.

Usage:
    python main.py
    python main.py --models "anthropic/claude-sonnet-4" "openai/gpt-4o"
    python main.py --min-level 3 --max-level 15 --trials 5
    python main.py --resume results/autism_bench_results_260311_120000.json
"""

import os
import sys
import json
import argparse
from datetime import datetime

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(
        description="AutismBench -Constraint Satisfaction Benchmark for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          # Run with defaults
  python main.py --models "openai/gpt-4o" "openai/o3"    # Specific models
  python main.py --min-level 3 --max-level 10 --trials 5  # Quick test run
  python main.py --dry-run                                 # Preview tasks without API calls
        """,
    )
    
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model IDs to benchmark (default: DEFAULT_MODELS from model_list.py)",
    )
    parser.add_argument(
        "--min-level", type=int, default=3,
        help="Minimum constraint level (default: 3)",
    )
    parser.add_argument(
        "--max-level", type=int, default=25,
        help="Maximum constraint level (default: 25)",
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Trials per level (default: 10)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--threads", type=int, default=10,
        help="Parallel API threads (default: 10)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: auto-generated)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from existing results file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate and display sample tasks without making API calls",
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not found.")
        print("Create a .env file with: OPENROUTER_API_KEY=your-key-here")
        sys.exit(1)
    
    # Resolve models
    if args.models:
        models = args.models
    else:
        from model_list import DEFAULT_MODELS
        models = DEFAULT_MODELS
    
    # Dry run: show sample tasks
    if args.dry_run:
        from autism_bench import generate_task
        print("\n" + "=" * 60)
        print("DRY RUN -Sample tasks at each level")
        print("=" * 60)
        
        for level in range(args.min_level, min(args.max_level + 1, args.min_level + 10)):
            task = generate_task(level, trial_seed=42)
            print(f"\n--- Level {level} ({len(task['constraints'])} constraints) ---")
            for i, ct in enumerate(task["constraint_texts"], 1):
                print(f"  {i}. {ct}")
        
        total_calls = len(models) * (args.max_level - args.min_level + 1) * args.trials
        print(f"\nWould run {total_calls} API calls across {len(models)} model(s)")
        print(f"Models: {', '.join(models)}")
        return
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        output_path = f"results/autism_bench_results_{ts}.json"
    
    # Run benchmark
    print("\n" + "=" * 60)
    print("  AutismBench v1.0 -The Constraint Tower")
    print("=" * 60)
    print(f"  Models:     {len(models)}")
    print(f"  Levels:     {args.min_level} → {args.max_level}")
    print(f"  Trials:     {args.trials} per level")
    print(f"  Temperature: {args.temperature}")
    print(f"  Threads:    {args.threads}")
    total_calls = len(models) * (args.max_level - args.min_level + 1) * args.trials
    print(f"  Total calls: {total_calls}")
    print(f"  Output:     {output_path}")
    print("=" * 60)
    
    from autism_bench import run_benchmark
    
    results = run_benchmark(
        api_key=api_key,
        models=models,
        min_level=args.min_level,
        max_level=args.max_level,
        trials_per_level=args.trials,
        temperature=args.temperature,
        max_threads=args.threads,
    )
    
    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total cost: ${results['stats']['total_cost_usd']:.4f}")
    print(f"Total time: {results['stats']['elapsed_seconds']:.0f}s")
    print(f"{'='*60}")
    
    # Print leaderboard
    print("\n  LEADERBOARD")
    print("  " + "-" * 50)
    ranked = sorted(
        results["models"].items(),
        key=lambda x: x[1]["total_score"],
        reverse=True,
    )
    for i, (model, data) in enumerate(ranked, 1):
        name = model.split("/")[-1] if "/" in model else model
        print(
            f"  #{i:2d}  {name:<30s}  "
            f"Score: {data['total_score']:>6d}  "
            f"Valid: {data['validity_ratio']:>5.1%}  "
            f"Perfect: {data['perfect_solve_rate']:>5.1%}"
        )


if __name__ == "__main__":
    main()

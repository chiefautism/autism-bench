"""
AutismBench -Core benchmark orchestration.

Generates tasks, runs them against models, collects and scores results.
"""

import json
import time
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from constraint_pool import select_constraints, format_constraint
from validator import extract_sentence, validate_task, score_results
from completions import CompletionClient
from utils import SYSTEM_PROMPT, build_prompt


def generate_task(level: int, trial_seed: int | None = None) -> dict:
    """
    Generate a single task with `level` constraints.
    
    Returns:
        {
            "level": int,
            "constraints": list[dict],
            "prompt": str,
            "constraint_texts": list[str],
        }
    """
    constraints = select_constraints(level, seed=trial_seed)
    constraint_texts = [format_constraint(c) for c in constraints]
    prompt = build_prompt(constraint_texts)
    
    return {
        "level": level,
        "constraints": constraints,
        "prompt": prompt,
        "constraint_texts": constraint_texts,
    }


def run_single_task(
    client: CompletionClient,
    model: str,
    task: dict,
    temperature: float = 0.0,
) -> dict:
    """
    Run a single task against a model and validate.
    
    Returns full result dict for this task.
    """
    # Call API
    completion = client.complete(
        model=model,
        prompt=task["prompt"],
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
    )
    
    # Extract sentence
    raw_response = completion["response"]
    sentence = extract_sentence(raw_response)
    
    # Validate
    results = validate_task(sentence, task["constraints"])
    scoring = score_results(results)
    
    return {
        "task": {
            "level": task["level"],
            "constraint_texts": task["constraint_texts"],
        },
        "raw_response": raw_response,
        "extracted_sentence": sentence,
        "results": [
            {
                "constraint_id": r["constraint_id"],
                "prompt_text": r["prompt_text"],
                "passed": r["passed"],
                "reason": r["reason"],
            }
            for r in results
        ],
        "score": scoring["score"],
        "passed": scoring["passed"],
        "total": scoring["total"],
        "perfect": scoring["perfect"],
        "latency_ms": completion["latency_ms"],
        "cost": completion["cost"],
        "error": completion.get("error"),
    }


def run_benchmark(
    api_key: str,
    models: list[str],
    min_level: int = 3,
    max_level: int = 25,
    trials_per_level: int = 10,
    temperature: float = 0.0,
    max_threads: int = 10,
    progress_callback=None,
) -> dict:
    """
    Run the full AutismBench benchmark.
    
    Args:
        api_key:            OpenRouter API key
        models:             List of model identifiers
        min_level:          Minimum constraint count
        max_level:          Maximum constraint count
        trials_per_level:   Random tasks per level
        temperature:        Sampling temperature
        max_threads:        Parallel API threads
        progress_callback:  Optional callable(model, level, trial, result)
    
    Returns:
        Full results dict ready for JSON serialization.
    """
    client = CompletionClient(api_key)
    
    total_levels = max_level - min_level + 1
    total_tasks = len(models) * total_levels * trials_per_level
    completed = 0
    
    all_results = {}
    
    start_time = time.time()
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        
        model_results = {
            "levels": {},
            "total_score": 0,
            "total_cost_usd": 0.0,
            "validity_ratio": 0.0,
            "perfect_solve_rate": 0.0,
        }
        
        total_passed = 0
        total_constraints = 0
        total_perfects = 0
        total_trials = 0
        
        for level in range(min_level, max_level + 1):
            level_results = []
            
            # Generate tasks
            tasks = []
            for trial in range(trials_per_level):
                seed = hash((model, level, trial)) % (2**31)
                task = generate_task(level, trial_seed=seed)
                tasks.append((trial, task))
            
            # Run tasks (can parallelize within a level)
            def _run_task(args):
                trial_idx, task = args
                result = run_single_task(client, model, task, temperature)
                return trial_idx, result
            
            with ThreadPoolExecutor(max_workers=min(max_threads, trials_per_level)) as executor:
                futures = {executor.submit(_run_task, t): t for t in tasks}
                for future in as_completed(futures):
                    trial_idx, result = future.result()
                    level_results.append(result)
                    
                    total_passed += result["passed"]
                    total_constraints += result["total"]
                    total_perfects += 1 if result["perfect"] else 0
                    total_trials += 1
                    
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(model, level, trial_idx, result)
            
            # Level summary
            level_scores = [r["score"] for r in level_results]
            level_perfects = sum(1 for r in level_results if r["perfect"])
            avg_score = sum(level_scores) / len(level_scores) if level_scores else 0
            
            model_results["levels"][str(level)] = {
                "avg_score": round(avg_score, 2),
                "max_score": max(level_scores) if level_scores else 0,
                "perfect_solves": level_perfects,
                "trials": level_results,
            }
            
            pct = (completed / total_tasks) * 100
            print(f"  Level {level:2d}: avg_score={avg_score:.1f}, "
                  f"perfect={level_perfects}/{trials_per_level}  "
                  f"[{pct:.0f}%]")
        
        # Model summary
        model_results["total_score"] = sum(
            r["score"]
            for lvl in model_results["levels"].values()
            for r in lvl["trials"]
        )
        model_results["validity_ratio"] = round(
            total_passed / total_constraints if total_constraints else 0, 4
        )
        model_results["perfect_solve_rate"] = round(
            total_perfects / total_trials if total_trials else 0, 4
        )
        model_results["total_cost_usd"] = round(
            sum(r["cost"] for lvl in model_results["levels"].values() for r in lvl["trials"]), 4
        )
        
        all_results[model] = model_results
        
        print(f"\n  TOTAL SCORE: {model_results['total_score']}")
        print(f"  Validity:    {model_results['validity_ratio']:.1%}")
        print(f"  Perfect:     {model_results['perfect_solve_rate']:.1%}")
        print(f"  Cost:        ${model_results['total_cost_usd']:.4f}")
    
    elapsed = time.time() - start_time
    stats = client.get_stats()
    
    return {
        "benchmark": "AutismBench",
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "min_level": min_level,
            "max_level": max_level,
            "trials_per_level": trials_per_level,
            "temperature": temperature,
        },
        "stats": {
            "total_cost_usd": stats["total_cost_usd"],
            "total_api_calls": stats["total_calls"],
            "elapsed_seconds": round(elapsed, 1),
        },
        "models": all_results,
    }

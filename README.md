# AutismBench

> **Created by the Chief of Autism.**
> It tests what autists can -and most models can't.

**AutismBench** is a lightweight, cheap-to-run benchmark for large language models that stress-tests
**simultaneous constraint satisfaction**, **precision**, **forward planning**, and **zero-tolerance rule adherence**.

The model must write a **single sentence** that satisfies **N constraints at once**.
Start at 3 -scale to 30+.

---

## Why "AutismBench"?

Created by the Chief of Autism. The benchmark is designed around cognitive strengths commonly associated with autism:

- **Pattern recognition under constraints** -seeing how 20 rules interact and finding the one solution that satisfies all of them
- **Hyperfocus on detail** -one wrong letter, one extra word, one broken rule = total failure
- **Systematizing** -building a mental system of interlocking rules and working within it precisely
- **Rule adherence without drift** -no "close enough", no rounding off, no approximation
- **Parallel constraint tracking** -holding many rules in working memory at once without losing any

AutismBench is a world of **explicit rules, zero ambiguity, and pure logic** -an environment where the autistic cognitive style is the gold standard.

**"Can your model think like an autist?"** If it can't hold 20 explicit rules without dropping one, it fails.

---

## How It Works

### The Metaphor

Think of a juggler. Three balls -easy. Five -fine. Ten -that's a circus act.
Twenty -you're either a savant or everything hits the floor.

### Example (Level 5)

The model receives:

```
Write a single sentence that satisfies ALL of the following constraints:

1. Contains exactly 8 words
2. Every word starts with a unique letter
3. The 3rd word is a past-tense verb
4. Contains exactly one number (as a digit)
5. The last word is an animal name
```

Valid answer: `"Big cats devoured 5 exotic fish near eagle"`

Every constraint is verifiable **programmatically** -no LLM judge needed.

### Difficulty Scaling

| Level | Constraints | Difficulty |
|-------|------------|------------|
| 3-5   | Basic      | Warm-up -most models pass |
| 6-10  | Medium     | Non-reasoning models start failing |
| 11-15 | Hard       | Only strong reasoning models survive |
| 16-20 | Expert     | Frontier reasoning models only |
| 21-30 | Insane     | Approaching NP-hard constraint satisfaction |

At level 25+, the task resembles a **constraint satisfaction problem** -there's no skill ceiling.

---

## Constraint Categories

### Structural
- Sentence contains exactly N words
- Word at position K has exactly M characters
- Total character count (excluding spaces) equals X

### Lexical
- Sentence must contain a color name
- Sentence must contain an animal name
- Sentence must contain a number as a digit
- Sentence must contain a profession/occupation
- Word at position K must be an adjective

### Positional
- First word must have exactly N letters
- Last word must end with suffix "-tion" / "-ing" / "-ly"
- Word at position K must be a verb in past tense
- The longest word must be at position K

### Relational
- Each word must start with a unique letter
- Each successive word must be longer than the previous
- Word at position K must rhyme with word at position J
- First letters of all words spell a valid English word (acrostic)

### Meta-structural
- Sum of all word lengths equals X
- Number of vowels in the sentence equals X
- Sentence must be a valid pangram subset (contain at least N unique letters)

---

## Scoring

**Per task (N constraints):**
- **+1 point** per satisfied constraint
- **×2 bonus** if ALL constraints are satisfied (perfect solve)

**Per level:** 10 random constraint combinations, averaged.

**Final score:** Sum across all levels.

**Validity ratio:** (constraints passed / constraints attempted) -exposes hallucinating models.

---

## Why This Is a Good Benchmark

| Property | AutismBench |
|----------|-------------|
| **Objectively verifiable** | Every constraint = binary pass/fail via script |
| **No LLM judge** | Pure code verification, zero judge bias |
| **No ceiling** | Add constraints infinitely, combinatorial explosion |
| **Cheap** | ~500 API calls per model, $2-5 total |
| **Tests what matters** | Planning, working memory, precision, constraint satisfaction |
| **Contamination-resistant** | Random constraint combos = infinite unique tasks |

---

## Repository Structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point -orchestrates everything |
| `autism_bench.py` | Core benchmark: generate tasks, run, score |
| `constraint_pool.py` | All constraints with validators and difficulty ratings |
| `validator.py` | Validation engine -checks each constraint |
| `completions.py` | Thread-safe API client (OpenRouter) with cost tracking |
| `model_list.py` | Supported model identifiers |
| `visualization.py` | Leaderboard plots and analysis |
| `utils.py` | Prompts, helpers, word lists |
| `requirements.txt` | Dependencies |

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/autism-bench.git
cd autism-bench

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create .env with your API key
echo "OPENROUTER_API_KEY=your-key-here" > .env

# Run benchmark
python main.py

# Specific models
python main.py --models "anthropic/claude-sonnet-4" "openai/o3"

# Custom levels
python main.py --min-level 3 --max-level 15 --trials 5

# Visualize results
python visualization.py results/autism_bench_results_*.json
```

---

## Estimated Cost

| Models | Levels | Trials | API Calls | Est. Cost |
|--------|--------|--------|-----------|-----------|
| 1 | 3-30 | 10 | 280 | $1-3 |
| 10 | 3-30 | 10 | 2,800 | $15-30 |
| 50 | 3-30 | 10 | 14,000 | $100-250 |

---

## Implementation Spec

See [TECH_TASK.md](TECH_TASK.md) for the full spec.

---

## License & Attribution

When publishing results or building on AutismBench:
1. **Credit the creator** -Chief of Autism
2. **Link to this repository**

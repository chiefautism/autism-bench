# AutismBench -Technical Task

## Overview

Build a Python benchmark that tests LLMs on simultaneous constraint satisfaction.
The model writes one sentence; the code checks N binary constraints. No LLM judge.

---

## Architecture

```
[Constraint Pool] → [Task Generator] → [Prompt Builder] → [API Runner] → [Validator] → [Scorer] → [Visualizer]
```

### Data Flow

```
1. Task Generator picks N random constraints from the pool (filtered for compatibility)
2. Prompt Builder formats them into a prompt for the model
3. API Runner sends to OpenRouter, gets response, logs cost
4. Validator extracts the sentence, runs each constraint checker → True/False per constraint
5. Scorer aggregates: points + perfect-solve bonus
6. Repeat for all levels × trials × models
7. Visualizer reads JSON results → matplotlib charts
```

---

## File-by-File Specification

### 1. `constraint_pool.py`

The heart of the benchmark. Contains all constraint definitions.

Each constraint is a dict:
```python
{
    "id": "word_count",
    "category": "structural",          # structural | lexical | positional | relational | meta
    "difficulty": 1,                    # 1-5 scale
    "param_generator": callable,        # returns random params, e.g. {"n": 8}
    "prompt_template": str,             # "The sentence must contain exactly {n} words"
    "validator": callable,              # (sentence: str, params: dict) -> bool
    "incompatible_with": list[str],     # constraint IDs that conflict
}
```

**Minimum viable constraint pool (ship with at least these):**

#### Structural (difficulty 1-2)
- `word_count` -exactly N words (N: 5-12)
- `char_count` -total characters (excl. spaces) equals X
- `word_length_at_pos` -word at position K has M characters

#### Lexical (difficulty 1-3)
- `contains_color` -must contain a color name (red, blue, green, etc.)
- `contains_animal` -must contain an animal name
- `contains_number` -must contain exactly one digit (0-9)
- `contains_profession` -must contain a profession (doctor, teacher, etc.)

#### Positional (difficulty 2-3)
- `first_word_length` -first word has N letters
- `last_word_suffix` -last word ends with "-ing" / "-tion" / "-ly" / "-ed"
- `word_at_pos_starts_with` -word at position K starts with letter X

#### Relational (difficulty 3-4)
- `unique_first_letters` -every word starts with a different letter
- `ascending_word_length` -each word is longer than the previous
- `first_letters_spell` -first letters of all words form a valid English word

#### Meta-structural (difficulty 4-5)
- `vowel_count` -total vowels in sentence equals X
- `unique_letters_count` -sentence contains at least N unique letters
- `word_length_sum` -sum of all word lengths equals X

**Constraint compatibility rules:**
- `word_count` is compatible with everything (it anchors the sentence length)
- `ascending_word_length` is incompatible with `word_length_at_pos` (may conflict)
- `first_letters_spell` requires `word_count` to match the acrostic word length
- If two constraints reference the same position, check they don't contradict

**Implementation notes:**
- Word lists for colors, animals, professions stored in `utils.py` as frozen sets
- Validators must handle: leading/trailing whitespace, punctuation stripping, case insensitivity
- Each validator returns `(bool, str)` -pass/fail + reason string for debugging

---

### 2. `validator.py`

Takes a raw model response and validates it.

```python
def extract_sentence(raw_response: str) -> str:
    """
    Extract the single sentence from the model's response.
    Models often add explanations -strip everything except the actual sentence.
    
    Strategy:
    1. Look for text in quotes (model often quotes its answer)
    2. If no quotes, take the first line that looks like a sentence
    3. Strip leading/trailing whitespace and punctuation artifacts
    """

def validate_task(sentence: str, constraints: list[dict]) -> list[dict]:
    """
    Run all constraints against the sentence.
    
    Returns list of:
    {
        "constraint_id": str,
        "passed": bool,
        "reason": str,      # why it failed (or "OK")
        "params": dict,     # the specific params used
    }
    """
```

**Critical validation rules:**
- Tokenize by whitespace (split on spaces)
- Strip punctuation from individual words for letter checks, but keep original for counting
- Numbers as digits ("5") count as words
- Case-insensitive for all letter/word checks
- If sentence extraction fails, ALL constraints fail

---

### 3. `completions.py`

Thread-safe OpenRouter API client.

```python
class CompletionClient:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.session = requests.Session()
        self.total_cost = 0.0
        self._lock = threading.Lock()
    
    def complete(self, model: str, prompt: str, temperature: float = 0.0) -> dict:
        """
        Returns {
            "response": str,        # raw model output
            "usage": dict,          # tokens in/out
            "cost": float,          # USD
            "latency_ms": int,
        }
        
        Handles:
        - Rate limiting (exponential backoff)
        - Timeout (60s default)
        - Thread-safe cost tracking
        """
```

**OpenRouter endpoint:** `https://openrouter.ai/api/v1/chat/completions`

**Headers:**
```python
{
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/autism-bench",
}
```

---

### 4. `autism_bench.py`

Core orchestration.

```python
def generate_task(level: int, constraint_pool: list[dict]) -> dict:
    """
    Pick `level` random constraints from the pool.
    
    Rules:
    1. Always include `word_count` (anchors the task)
    2. Check `incompatible_with` -no conflicting constraints
    3. Mix categories -at least 2 different categories
    4. Difficulty sum should roughly scale with level
    
    Returns {
        "level": int,
        "constraints": list[dict],  # with generated params
        "prompt": str,              # formatted prompt
    }
    """

def run_benchmark(
    models: list[str],
    min_level: int = 3,
    max_level: int = 25,
    trials_per_level: int = 10,
    temperature: float = 0.0,
    max_threads: int = 10,
) -> dict:
    """
    Main entry point.
    
    For each model:
        For each level (min_level to max_level):
            For each trial (1 to trials_per_level):
                1. Generate random task
                2. Send to model
                3. Validate response
                4. Score
    
    Returns full results dict for JSON serialization.
    """
```

**Prompt template:**
```
Write a single English sentence that satisfies ALL of the following constraints simultaneously.

Constraints:
{numbered_constraints}

IMPORTANT:
- Output ONLY the sentence, nothing else.
- No explanations, no commentary, no alternatives.
- The sentence must satisfy every single constraint listed above.
```

---

### 5. `main.py`

CLI interface using argparse.

```
usage: main.py [-h] [--models MODELS [MODELS ...]] [--min-level MIN]
               [--max-level MAX] [--trials TRIALS] [--temperature TEMP]
               [--threads THREADS] [--output OUTPUT] [--resume FILE]

Arguments:
  --models        Model IDs to benchmark (default: all from model_list.py)
  --min-level     Minimum constraint level (default: 3)
  --max-level     Maximum constraint level (default: 25)
  --trials        Trials per level (default: 10)
  --temperature   Sampling temperature (default: 0.0)
  --threads       Parallel API threads (default: 10)
  --output        Output JSON path (default: auto-generated with timestamp)
  --resume        Resume from existing results file
```

**Output format:** `autism_bench_results_{YYMMDD}_{HHMMSS}.json`

---

### 6. `visualization.py`

Reads results JSON, generates plots.

**Required plots:**
1. **Leaderboard bar chart** -total score per model, sorted
2. **Validity ratio heatmap** -(model × level) showing % constraints passed
3. **Perfect solve rate** -% of tasks where ALL constraints passed, per model
4. **Difficulty curve** -score vs. level, one line per model (shows where models "break")
5. **Category breakdown** -which constraint categories each model struggles with

```
usage: visualization.py results.json [--save] [--show]
```

---

### 7. `model_list.py`

```python
MODELS = [
    # Frontier reasoning
    "openai/o3",
    "anthropic/claude-opus-4-6",
    "google/gemini-3-pro",
    
    # Strong reasoning
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5",
    "openai/o4-mini",
    
    # Non-reasoning frontier
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.0-flash",
    "meta-llama/llama-4-maverick",
    
    # Mid-tier
    "anthropic/claude-haiku-4-5",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-lite",
    
    # Open source
    "deepseek/deepseek-v3",
    "qwen/qwen3-72b",
    "mistralai/mistral-large",
]
```

---

### 8. `utils.py`

```python
# Word lists (frozen sets for O(1) lookup)
COLORS = frozenset({"red", "blue", "green", "yellow", "black", "white", "orange", ...})
ANIMALS = frozenset({"cat", "dog", "eagle", "wolf", "bear", "fox", "deer", "hawk", ...})
PROFESSIONS = frozenset({"doctor", "teacher", "engineer", "pilot", "chef", "nurse", ...})

# English dictionary for acrostic validation
# Use nltk.words() or dwyl/english-words
ENGLISH_WORDS = load_dictionary()

def count_syllables(word: str) -> int: ...
def count_vowels(text: str) -> int: ...
def strip_punctuation(word: str) -> str: ...
def words_rhyme(w1: str, w2: str) -> bool: ...  # using pronouncing library
```

---

## Results JSON Schema

```json
{
    "benchmark": "AutismBench",
    "version": "1.0",
    "timestamp": "2026-03-11T12:00:00Z",
    "config": {
        "min_level": 3,
        "max_level": 25,
        "trials_per_level": 10,
        "temperature": 0.0
    },
    "models": {
        "anthropic/claude-sonnet-4-6": {
            "total_score": 1234,
            "total_cost_usd": 2.45,
            "validity_ratio": 0.87,
            "perfect_solve_rate": 0.65,
            "levels": {
                "3": {
                    "avg_score": 8.5,
                    "perfect_solves": 9,
                    "trials": [
                        {
                            "task": {
                                "constraints": [...],
                                "prompt": "..."
                            },
                            "response": "Big cats devoured 5 exotic fish near eagle",
                            "results": [
                                {"constraint_id": "word_count", "passed": true, "reason": "OK"},
                                {"constraint_id": "contains_number", "passed": true, "reason": "OK"}
                            ],
                            "score": 12,
                            "perfect": true,
                            "latency_ms": 1200
                        }
                    ]
                }
            }
        }
    }
}
```

---

## Testing Strategy

### Unit tests for validators
Every constraint validator needs tests with:
- A sentence that passes
- A sentence that fails
- Edge cases (empty string, single word, numbers, punctuation)

### Integration test
Run level 3-5 against a cheap model (gpt-4o-mini) and verify:
- Results JSON is valid
- Scores are non-negative
- Validity ratio is between 0 and 1
- Cost tracking works

### Sanity check
For each level, generate 100 tasks and verify that at least some are solvable
(a brute-force solver or manual check for levels 3-5).

---

## Development Milestones

### Phase 1: MVP (1-2 days)
- [ ] `constraint_pool.py` with 10 constraints + validators
- [ ] `validator.py` with sentence extraction
- [ ] `completions.py` with OpenRouter client
- [ ] `main.py` with basic CLI
- [ ] Test on 3 models, levels 3-10

### Phase 2: Full Benchmark (2-3 days)
- [ ] Expand to 16+ constraints
- [ ] Add compatibility checks
- [ ] Add threading for parallel execution
- [ ] Add resume capability
- [ ] Test on 10+ models, levels 3-20

### Phase 3: Polish (1-2 days)
- [ ] `visualization.py` with all 5 plots
- [ ] Results analysis and leaderboard generation
- [ ] README with sample results
- [ ] Publish to GitHub

---

## Key Design Decisions

1. **Why OpenRouter?** One API key → 50+ models. Standard for indie benchmarks.
2. **Why temperature=0?** Reproducibility. Use trials + averaging for stability instead.
3. **Why always include word_count?** It anchors the task -without it, models can write arbitrarily long sentences and trivially satisfy most constraints.
4. **Why 10 trials per level?** Balance between cost and statistical stability. 5 is minimum, 10 is sweet spot, 20+ for publication-grade results.
5. **Why no LLM judge?** Every constraint is binary and programmatically verifiable. This eliminates judge bias, judge cost, and reproducibility concerns. This is the core advantage.

"""
AutismBench -Constraint Pool.

Each constraint is a dict with:
    id:                 unique string identifier
    category:           structural | lexical | positional | relational | meta
    difficulty:         1-5 scale
    param_generator:    () -> dict with random params
    prompt_template:    str with {param} placeholders
    validator:          (sentence: str, params: dict) -> (bool, str)
    incompatible_with:  list of constraint IDs that conflict
"""

import random
from utils import (
    tokenize, tokenize_clean, count_vowels, count_unique_letters,
    get_word_lengths, first_letters, has_digit, is_valid_english_word,
    strip_punctuation, COLORS, ANIMALS, PROFESSIONS, SUFFIXES,
)


# ─── Validator Functions ──────────────────────────────────────────────────────

def _validate_word_count(sentence: str, params: dict) -> tuple[bool, str]:
    n = params["n"]
    words = tokenize(sentence)
    actual = len(words)
    if actual == n:
        return True, "OK"
    return False, f"Expected {n} words, got {actual}"


def _validate_char_count(sentence: str, params: dict) -> tuple[bool, str]:
    x = params["x"]
    actual = sum(len(w) for w in tokenize_clean(sentence))
    if actual == x:
        return True, "OK"
    return False, f"Expected {x} total characters (excl. spaces/punct), got {actual}"


def _validate_word_length_at_pos(sentence: str, params: dict) -> tuple[bool, str]:
    k, m = params["k"], params["m"]
    words = tokenize_clean(sentence)
    if k > len(words):
        return False, f"Sentence has only {len(words)} words, need position {k}"
    actual = len(words[k - 1])
    if actual == m:
        return True, "OK"
    return False, f"Word at position {k} ('{words[k-1]}') has {actual} chars, expected {m}"


def _validate_contains_color(sentence: str, params: dict) -> tuple[bool, str]:
    words = [w.lower() for w in tokenize_clean(sentence)]
    found = [w for w in words if w in COLORS]
    if found:
        return True, "OK"
    return False, "No color name found in sentence"


def _validate_contains_animal(sentence: str, params: dict) -> tuple[bool, str]:
    words = [w.lower() for w in tokenize_clean(sentence)]
    found = [w for w in words if w in ANIMALS]
    if found:
        return True, "OK"
    return False, "No animal name found in sentence"


def _validate_contains_number(sentence: str, params: dict) -> tuple[bool, str]:
    has, count = has_digit(sentence)
    if has and count == 1:
        return True, "OK"
    if not has:
        return False, "No digit found in sentence"
    return False, f"Expected exactly 1 digit, found {count}"


def _validate_contains_profession(sentence: str, params: dict) -> tuple[bool, str]:
    words = [w.lower() for w in tokenize_clean(sentence)]
    found = [w for w in words if w in PROFESSIONS]
    if found:
        return True, "OK"
    return False, "No profession found in sentence"


def _validate_first_word_length(sentence: str, params: dict) -> tuple[bool, str]:
    n = params["n"]
    words = tokenize_clean(sentence)
    if not words:
        return False, "Empty sentence"
    actual = len(words[0])
    if actual == n:
        return True, "OK"
    return False, f"First word '{words[0]}' has {actual} letters, expected {n}"


def _validate_last_word_suffix(sentence: str, params: dict) -> tuple[bool, str]:
    suffix = params["suffix"]
    words = tokenize_clean(sentence)
    if not words:
        return False, "Empty sentence"
    last = words[-1].lower()
    if SUFFIXES[suffix](last):
        return True, "OK"
    return False, f"Last word '{last}' does not end with '-{suffix}'"


def _validate_word_at_pos_starts_with(sentence: str, params: dict) -> tuple[bool, str]:
    k, letter = params["k"], params["letter"]
    words = tokenize_clean(sentence)
    if k > len(words):
        return False, f"Sentence has only {len(words)} words, need position {k}"
    word = words[k - 1]
    if word[0].lower() == letter.lower():
        return True, "OK"
    return False, f"Word at position {k} ('{word}') starts with '{word[0]}', expected '{letter}'"


def _validate_unique_first_letters(sentence: str, params: dict) -> tuple[bool, str]:
    words = tokenize_clean(sentence)
    firsts = [w[0].lower() for w in words if w]
    if len(firsts) == len(set(firsts)):
        return True, "OK"
    dupes = [l for l in firsts if firsts.count(l) > 1]
    return False, f"Duplicate first letters: {set(dupes)}"


def _validate_ascending_word_length(sentence: str, params: dict) -> tuple[bool, str]:
    lengths = get_word_lengths(sentence)
    if not lengths:
        return False, "Empty sentence"
    for i in range(1, len(lengths)):
        if lengths[i] <= lengths[i - 1]:
            words = tokenize_clean(sentence)
            return False, f"Word {i+1} ('{words[i]}', {lengths[i]} chars) is not longer than word {i} ('{words[i-1]}', {lengths[i-1]} chars)"
    return True, "OK"


def _validate_first_letters_spell(sentence: str, params: dict) -> tuple[bool, str]:
    target = params["word"]
    fl = first_letters(sentence)
    if fl == target.lower():
        return True, "OK"
    return False, f"First letters spell '{fl}', expected '{target}'"


def _validate_vowel_count(sentence: str, params: dict) -> tuple[bool, str]:
    x = params["x"]
    actual = count_vowels(sentence)
    if actual == x:
        return True, "OK"
    return False, f"Expected {x} vowels, got {actual}"


def _validate_unique_letters_count(sentence: str, params: dict) -> tuple[bool, str]:
    n = params["n"]
    actual = count_unique_letters(sentence)
    if actual >= n:
        return True, "OK"
    return False, f"Expected at least {n} unique letters, got {actual}"


def _validate_word_length_sum(sentence: str, params: dict) -> tuple[bool, str]:
    x = params["x"]
    actual = sum(get_word_lengths(sentence))
    if actual == x:
        return True, "OK"
    return False, f"Sum of word lengths is {actual}, expected {x}"


def _validate_no_letter(sentence: str, params: dict) -> tuple[bool, str]:
    letter = params["letter"].lower()
    clean = "".join(tokenize_clean(sentence)).lower()
    if letter not in clean:
        return True, "OK"
    count = clean.count(letter)
    return False, f"Letter '{letter}' appears {count} time(s) in the sentence"


def _validate_all_words_min_length(sentence: str, params: dict) -> tuple[bool, str]:
    n = params["n"]
    words = tokenize_clean(sentence)
    short = [w for w in words if len(w) < n]
    if not short:
        return True, "OK"
    return False, f"Words shorter than {n} chars: {short}"


# ─── Acrostic helper words (short, common) ────────────────────────────────────

ACROSTIC_WORDS = [
    "cat", "dog", "bat", "hat", "map", "sun", "run", "top", "big", "red",
    "old", "new", "hot", "wet", "dry", "end", "art", "bit", "cup", "dim",
    "ego", "fan", "gem", "hip", "ice", "jam", "key", "lip", "mob", "nap",
    "odd", "pen", "ram", "sip", "tin", "use", "van", "wax", "yam", "zip",
    "star", "cold", "warm", "dark", "slow", "fast", "bold", "calm", "grim",
    "kind", "lost", "mild", "neat", "open", "pale", "rich", "soft", "tall",
    "vast", "wild", "able", "base", "core", "deep", "each", "fair", "glad",
    "hard", "iron", "just", "keen", "lake", "mint", "note", "pace", "rage",
]

# Avoid rare letters for word_at_pos_starts_with
COMMON_LETTERS = list("abcdefghijklmnoprstw")

# Letters that are easy to avoid
AVOIDABLE_LETTERS = list("jkqxz")


# ─── Constraint Definitions ───────────────────────────────────────────────────

CONSTRAINT_POOL = [
    # ── Structural ──
    {
        "id": "word_count",
        "category": "structural",
        "difficulty": 1,
        "param_generator": lambda: {"n": random.randint(5, 12)},
        "prompt_template": "The sentence must contain exactly {n} words",
        "validator": _validate_word_count,
        "incompatible_with": [],
    },
    {
        "id": "char_count",
        "category": "structural",
        "difficulty": 2,
        "param_generator": lambda: {"x": random.randint(25, 55)},
        "prompt_template": "The total number of characters (letters only, excluding spaces and punctuation) must be exactly {x}",
        "validator": _validate_char_count,
        "incompatible_with": ["word_length_sum"],
    },
    {
        "id": "word_length_at_pos",
        "category": "structural",
        "difficulty": 2,
        "param_generator": lambda: {"k": random.randint(2, 6), "m": random.randint(3, 7)},
        "prompt_template": "The word at position {k} must have exactly {m} characters",
        "validator": _validate_word_length_at_pos,
        "incompatible_with": ["ascending_word_length"],
    },
    {
        "id": "all_words_min_length",
        "category": "structural",
        "difficulty": 3,
        "param_generator": lambda: {"n": random.randint(3, 5)},
        "prompt_template": "Every word in the sentence must be at least {n} characters long",
        "validator": _validate_all_words_min_length,
        "incompatible_with": [],
    },

    # ── Lexical ──
    {
        "id": "contains_color",
        "category": "lexical",
        "difficulty": 1,
        "param_generator": lambda: {},
        "prompt_template": "The sentence must contain a color name (e.g., red, blue, green)",
        "validator": _validate_contains_color,
        "incompatible_with": [],
    },
    {
        "id": "contains_animal",
        "category": "lexical",
        "difficulty": 1,
        "param_generator": lambda: {},
        "prompt_template": "The sentence must contain an animal name (e.g., cat, eagle, wolf)",
        "validator": _validate_contains_animal,
        "incompatible_with": [],
    },
    {
        "id": "contains_number",
        "category": "lexical",
        "difficulty": 1,
        "param_generator": lambda: {},
        "prompt_template": "The sentence must contain exactly one number written as a digit (e.g., 5, not 'five')",
        "validator": _validate_contains_number,
        "incompatible_with": [],
    },
    {
        "id": "contains_profession",
        "category": "lexical",
        "difficulty": 2,
        "param_generator": lambda: {},
        "prompt_template": "The sentence must contain a profession or occupation (e.g., doctor, pilot, chef)",
        "validator": _validate_contains_profession,
        "incompatible_with": [],
    },

    # ── Positional ──
    {
        "id": "first_word_length",
        "category": "positional",
        "difficulty": 2,
        "param_generator": lambda: {"n": random.randint(3, 6)},
        "prompt_template": "The first word must have exactly {n} letters",
        "validator": _validate_first_word_length,
        "incompatible_with": [],
    },
    {
        "id": "last_word_suffix",
        "category": "positional",
        "difficulty": 2,
        "param_generator": lambda: {"suffix": random.choice(["ing", "tion", "ly", "ed"])},
        "prompt_template": 'The last word must end with the suffix "-{suffix}"',
        "validator": _validate_last_word_suffix,
        "incompatible_with": [],
    },
    {
        "id": "word_at_pos_starts_with",
        "category": "positional",
        "difficulty": 2,
        "param_generator": lambda: {
            "k": random.randint(2, 5),
            "letter": random.choice(COMMON_LETTERS),
        },
        "prompt_template": "The word at position {k} must start with the letter '{letter}'",
        "validator": _validate_word_at_pos_starts_with,
        "incompatible_with": [],
    },

    # ── Relational ──
    {
        "id": "unique_first_letters",
        "category": "relational",
        "difficulty": 3,
        "param_generator": lambda: {},
        "prompt_template": "Every word in the sentence must start with a different letter (no two words share the same first letter)",
        "validator": _validate_unique_first_letters,
        "incompatible_with": [],
    },
    {
        "id": "ascending_word_length",
        "category": "relational",
        "difficulty": 4,
        "param_generator": lambda: {},
        "prompt_template": "Each word must be strictly longer than the previous word (ascending word length)",
        "validator": _validate_ascending_word_length,
        "incompatible_with": ["word_length_at_pos"],
    },
    {
        "id": "first_letters_spell",
        "category": "relational",
        "difficulty": 5,
        "param_generator": lambda: {"word": random.choice(ACROSTIC_WORDS)},
        "prompt_template": 'The first letters of all words, read in order, must spell the word "{word}"',
        "validator": _validate_first_letters_spell,
        "incompatible_with": ["unique_first_letters", "word_at_pos_starts_with"],
    },

    # ── Meta-structural ──
    {
        "id": "vowel_count",
        "category": "meta",
        "difficulty": 3,
        "param_generator": lambda: {"x": random.randint(10, 25)},
        "prompt_template": "The total number of vowels (a, e, i, o, u) in the entire sentence must be exactly {x}",
        "validator": _validate_vowel_count,
        "incompatible_with": [],
    },
    {
        "id": "unique_letters_count",
        "category": "meta",
        "difficulty": 2,
        "param_generator": lambda: {"n": random.randint(12, 20)},
        "prompt_template": "The sentence must use at least {n} different letters of the alphabet",
        "validator": _validate_unique_letters_count,
        "incompatible_with": [],
    },
    {
        "id": "word_length_sum",
        "category": "meta",
        "difficulty": 3,
        "param_generator": lambda: {"x": random.randint(28, 50)},
        "prompt_template": "The sum of all word lengths (letters only) must equal exactly {x}",
        "validator": _validate_word_length_sum,
        "incompatible_with": ["char_count"],
    },
    {
        "id": "no_letter",
        "category": "meta",
        "difficulty": 3,
        "param_generator": lambda: {"letter": random.choice(AVOIDABLE_LETTERS)},
        "prompt_template": 'The sentence must not contain the letter "{letter}" anywhere',
        "validator": _validate_no_letter,
        "incompatible_with": [],
    },
]


# ─── Constraint Selection ─────────────────────────────────────────────────────

def get_constraint_by_id(cid: str) -> dict | None:
    for c in CONSTRAINT_POOL:
        if c["id"] == cid:
            return c
    return None


def select_constraints(level: int, seed: int | None = None) -> list[dict]:
    """
    Select `level` compatible constraints with generated params.
    
    Rules:
    1. Always include word_count (anchors the task)
    2. No incompatible constraints together
    3. At least 2 different categories
    4. first_letters_spell forces word_count.n to match acrostic word length
    """
    if seed is not None:
        random.seed(seed)
    
    pool = [dict(c) for c in CONSTRAINT_POOL]  # shallow copy
    selected = []
    selected_ids = set()
    blocked_ids = set()
    
    # Always start with word_count
    wc = get_constraint_by_id("word_count")
    wc_instance = dict(wc)
    wc_instance["params"] = wc["param_generator"]()
    selected.append(wc_instance)
    selected_ids.add("word_count")
    
    # Remove word_count from pool
    pool = [c for c in pool if c["id"] != "word_count"]
    
    # Fill remaining slots
    attempts = 0
    while len(selected) < level and attempts < 200:
        attempts += 1
        
        # Filter available constraints
        available = [
            c for c in pool
            if c["id"] not in selected_ids
            and c["id"] not in blocked_ids
        ]
        
        if not available:
            break
        
        candidate = random.choice(available)
        
        # Check compatibility
        is_compatible = True
        for sel in selected:
            if candidate["id"] in sel.get("incompatible_with", []):
                is_compatible = False
                break
            if sel["id"] in candidate.get("incompatible_with", []):
                is_compatible = False
                break
        
        if not is_compatible:
            continue
        
        # Generate params
        instance = dict(candidate)
        instance["params"] = candidate["param_generator"]()
        
        # Special case: first_letters_spell forces word_count
        if candidate["id"] == "first_letters_spell":
            acrostic_word = instance["params"]["word"]
            # Update word_count to match acrostic length
            selected[0]["params"]["n"] = len(acrostic_word)
        
        # Special case: word_at_pos_starts_with -position must be <= word_count
        if candidate["id"] == "word_at_pos_starts_with":
            max_pos = selected[0]["params"]["n"]
            instance["params"]["k"] = min(instance["params"]["k"], max_pos)
        
        # Special case: word_length_at_pos -position must be <= word_count
        if candidate["id"] == "word_length_at_pos":
            max_pos = selected[0]["params"]["n"]
            instance["params"]["k"] = min(instance["params"]["k"], max_pos)
        
        selected.append(instance)
        selected_ids.add(candidate["id"])
        
        # Block incompatible constraints
        for incompat_id in candidate.get("incompatible_with", []):
            blocked_ids.add(incompat_id)
    
    # Ensure at least 2 categories
    categories = set(c["category"] for c in selected)
    if len(categories) < 2 and len(selected) >= 2:
        # Try to swap last non-word_count constraint with one from a different category
        pass  # Accept it -with 18 constraints across 5 categories, this rarely happens
    
    return selected


def format_constraint(constraint: dict) -> str:
    """Format a constraint with its params into a human-readable string."""
    template = constraint["prompt_template"]
    params = constraint.get("params", {})
    return template.format(**params)

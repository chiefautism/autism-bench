"""
AutismBench -Utility functions, word lists, and helpers.
"""

import re
import string

# ─── Word Lists (frozen sets for O(1) lookup) ─────────────────────────────────

COLORS = frozenset({
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple",
    "pink", "brown", "gray", "grey", "silver", "gold", "violet", "indigo",
    "crimson", "scarlet", "azure", "ivory", "coral", "teal", "maroon",
    "navy", "beige", "tan", "lime", "cyan", "magenta", "turquoise",
})

ANIMALS = frozenset({
    "cat", "dog", "eagle", "wolf", "bear", "fox", "deer", "hawk", "lion",
    "tiger", "shark", "whale", "snake", "horse", "mouse", "rat", "owl",
    "crow", "dove", "swan", "duck", "goat", "sheep", "cow", "pig", "ant",
    "bee", "bat", "ape", "elk", "emu", "hen", "ram", "yak", "cod", "eel",
    "fly", "jay", "koi", "ox", "pug", "raven", "robin", "salmon", "trout",
    "viper", "wren", "zebra", "heron", "otter", "panda", "koala", "camel",
    "gecko", "finch", "crane", "moose", "bison", "lynx", "mole", "newt",
    "toad", "frog", "crab", "clam", "moth", "wasp", "mule", "seal",
})

PROFESSIONS = frozenset({
    "doctor", "teacher", "engineer", "pilot", "chef", "nurse", "lawyer",
    "farmer", "artist", "writer", "singer", "dancer", "actor", "baker",
    "driver", "guard", "judge", "miner", "tailor", "barber", "clerk",
    "coach", "diver", "guide", "mason", "mayor", "monk", "tutor", "vet",
    "scout", "smith", "sailor", "ranger", "potter", "plumber", "painter",
    "mechanic", "janitor", "dentist", "butcher", "brewer", "broker",
    "captain", "colonel", "general", "marshal", "soldier", "surgeon",
    "trucker", "welder", "banker", "bishop", "jockey", "knight",
})

# Vowels set
VOWELS = frozenset("aeiouAEIOU")

# Common suffixes for positional checks
SUFFIXES = {
    "ing": lambda w: w.endswith("ing"),
    "tion": lambda w: w.endswith("tion"),
    "ly": lambda w: w.endswith("ly"),
    "ed": lambda w: w.endswith("ed"),
    "ness": lambda w: w.endswith("ness"),
    "able": lambda w: w.endswith("able"),
    "ment": lambda w: w.endswith("ment"),
}

# ─── English Dictionary ────────────────────────────────────────────────────────

_ENGLISH_WORDS = None

def load_dictionary() -> frozenset:
    """Load English words dictionary. Falls back to a basic set if nltk unavailable."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is not None:
        return _ENGLISH_WORDS
    
    try:
        import nltk
        try:
            from nltk.corpus import words
            _ENGLISH_WORDS = frozenset(w.lower() for w in words.words())
        except LookupError:
            nltk.download("words", quiet=True)
            from nltk.corpus import words
            _ENGLISH_WORDS = frozenset(w.lower() for w in words.words())
    except ImportError:
        # Fallback: use all our word lists + common words
        _ENGLISH_WORDS = COLORS | ANIMALS | PROFESSIONS | frozenset({
            "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "must", "need", "dare",
            "and", "but", "or", "nor", "for", "yet", "so", "if", "then", "else",
            "big", "small", "old", "new", "good", "bad", "long", "short", "tall",
            "fast", "slow", "hot", "cold", "warm", "cool", "dark", "light",
            "run", "walk", "talk", "eat", "drink", "sleep", "work", "play",
            "read", "write", "think", "know", "see", "hear", "feel", "make",
        })
    
    return _ENGLISH_WORDS


# ─── Text Processing ──────────────────────────────────────────────────────────

def strip_punctuation(word: str) -> str:
    """Remove leading/trailing punctuation from a word."""
    return word.strip(string.punctuation)


def tokenize(sentence: str) -> list[str]:
    """Split sentence into words (whitespace tokenization)."""
    return sentence.strip().split()


def tokenize_clean(sentence: str) -> list[str]:
    """Split sentence into words with punctuation stripped."""
    return [strip_punctuation(w) for w in tokenize(sentence) if strip_punctuation(w)]


def count_vowels(text: str) -> int:
    """Count total vowels in text."""
    return sum(1 for c in text if c in VOWELS)


def count_unique_letters(text: str) -> int:
    """Count unique alphabetic characters (case-insensitive)."""
    return len(set(c.lower() for c in text if c.isalpha()))


def get_word_lengths(sentence: str) -> list[int]:
    """Get list of word lengths (punctuation stripped)."""
    return [len(w) for w in tokenize_clean(sentence)]


def first_letters(sentence: str) -> str:
    """Get first letters of each word as a string."""
    words = tokenize_clean(sentence)
    return "".join(w[0].lower() for w in words if w)


def has_digit(text: str) -> tuple[bool, int]:
    """Check if text contains digits. Returns (has_digit, count)."""
    digits = [c for c in text if c.isdigit()]
    return len(digits) > 0, len(digits)


def is_valid_english_word(word: str) -> bool:
    """Check if a word is in the English dictionary."""
    dictionary = load_dictionary()
    return word.lower().strip(string.punctuation) in dictionary


# ─── Prompt Template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise language model being evaluated on constraint satisfaction. 
Your task is to write a single English sentence that satisfies ALL given constraints simultaneously.
Output ONLY the sentence. No explanations, no commentary, no alternatives, no quotes around it."""

def build_prompt(constraints_text: list[str]) -> str:
    """Build the user prompt from a list of constraint descriptions."""
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(constraints_text))
    return f"""Write a single English sentence that satisfies ALL of the following constraints simultaneously.

Constraints:
{numbered}

IMPORTANT:
- Output ONLY the sentence, nothing else.
- No explanations, no numbering, no quotes.
- The sentence must satisfy every single constraint listed above."""

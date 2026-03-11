"""
AutismBench -Supported model identifiers (OpenRouter format).
"""

# Frontier reasoning models
REASONING_MODELS = [
    "openai/o3",
    "anthropic/claude-opus-4-6",
    "google/gemini-2.5-pro",
]

# Strong reasoning / thinking models
STRONG_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "openai/o4-mini",
    "openai/gpt-5",
    "deepseek/deepseek-r1",
]

# Non-reasoning frontier
FRONTIER_MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.0-flash",
    "meta-llama/llama-4-maverick",
]

# Mid-tier
MID_TIER_MODELS = [
    "anthropic/claude-haiku-4-5",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-lite",
]

# Open source
OPEN_SOURCE_MODELS = [
    "deepseek/deepseek-chat",
    "qwen/qwen3-72b",
    "mistralai/mistral-large",
]

# Default: a reasonable cross-section for quick benchmarking
DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash",
    "deepseek/deepseek-chat",
]

# All models
ALL_MODELS = (
    REASONING_MODELS
    + STRONG_MODELS
    + FRONTIER_MODELS
    + MID_TIER_MODELS
    + OPEN_SOURCE_MODELS
)

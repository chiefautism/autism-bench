"""
AutismBench -Validator.

Extracts the sentence from model response and validates each constraint.
"""

import re


def extract_sentence(raw_response: str) -> str:
    """
    Extract the single sentence from the model's response.
    
    Models often add explanations, quotes, or formatting.
    Strategy:
    1. If response is a single line, use it directly
    2. Look for text in quotes
    3. Take the first non-empty line that looks like a sentence
    4. Strip artifacts
    """
    if not raw_response or not raw_response.strip():
        return ""
    
    text = raw_response.strip()
    
    # If it's a single line (no newlines), use directly
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) == 1:
        return _clean_sentence(lines[0])
    
    # Look for quoted text (model often quotes its answer)
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        # Return the longest quoted string (likely the sentence)
        return _clean_sentence(max(quoted, key=len))
    
    # Look for text after common prefixes
    for prefix in ["Here is", "Here's", "Sentence:", "Answer:", "Output:"]:
        for line in lines:
            if line.lower().startswith(prefix.lower()):
                rest = line[len(prefix):].strip().lstrip(":").strip()
                if rest:
                    return _clean_sentence(rest)
    
    # Heuristic: take the first line that looks like a real sentence
    # (starts with uppercase or digit, has spaces, reasonable length)
    for line in lines:
        cleaned = _clean_sentence(line)
        words = cleaned.split()
        if len(words) >= 3 and (cleaned[0].isupper() or cleaned[0].isdigit()):
            return cleaned
    
    # Fallback: first non-empty line
    if lines:
        return _clean_sentence(lines[0])
    
    return ""


def _clean_sentence(s: str) -> str:
    """Clean up a sentence: remove surrounding quotes, extra whitespace."""
    s = s.strip()
    # Remove surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # Remove markdown bold/italic
    s = re.sub(r'\*+', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def validate_task(sentence: str, constraints: list[dict]) -> list[dict]:
    """
    Run all constraints against the sentence.
    
    Returns list of:
    {
        "constraint_id": str,
        "prompt_text": str,
        "passed": bool,
        "reason": str,
        "params": dict,
    }
    """
    from constraint_pool import format_constraint
    
    results = []
    
    if not sentence:
        # All constraints fail on empty sentence
        for c in constraints:
            results.append({
                "constraint_id": c["id"],
                "prompt_text": format_constraint(c),
                "passed": False,
                "reason": "Empty or unparseable sentence",
                "params": c.get("params", {}),
            })
        return results
    
    for c in constraints:
        params = c.get("params", {})
        try:
            passed, reason = c["validator"](sentence, params)
        except Exception as e:
            passed, reason = False, f"Validator error: {str(e)}"
        
        results.append({
            "constraint_id": c["id"],
            "prompt_text": format_constraint(c),
            "passed": passed,
            "reason": reason,
            "params": params,
        })
    
    return results


def score_results(results: list[dict]) -> dict:
    """
    Score a single task's results.
    
    Returns:
    {
        "points": int,          # raw points
        "passed": int,          # constraints passed
        "total": int,           # total constraints
        "perfect": bool,        # all passed?
        "score": int,           # final score (with bonus)
    }
    """
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    perfect = passed == total
    
    # Score: 1 point per constraint + 2x bonus for perfect
    points = passed
    score = points * 2 if perfect else points
    
    return {
        "points": points,
        "passed": passed,
        "total": total,
        "perfect": perfect,
        "score": score,
    }

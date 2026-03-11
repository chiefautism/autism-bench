"""
AutismBench -Thread-safe OpenRouter API client with cost tracking.
"""

import time
import threading
import requests


class CompletionClient:
    """Thread-safe OpenRouter completion client."""
    
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: str, max_retries: int = 3, timeout: int = 60):
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/autism-bench",
            "X-Title": "AutismBench",
        })
        
        self.total_cost = 0.0
        self.total_calls = 0
        self._lock = threading.Lock()
    
    def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> dict:
        """
        Send a completion request to OpenRouter.
        
        Returns:
            {
                "response": str,
                "usage": {"prompt_tokens": int, "completion_tokens": int},
                "cost": float,
                "latency_ms": int,
                "model": str,
            }
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 500,  # Single sentence shouldn't need more
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                resp = self.session.post(
                    self.OPENROUTER_URL,
                    json=payload,
                    timeout=self.timeout,
                )
                latency_ms = int((time.time() - start) * 1000)
                
                if resp.status_code == 429:
                    # Rate limited -back off
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited on {model}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                # Extract response
                response_text = ""
                if "choices" in data and data["choices"]:
                    response_text = data["choices"][0].get("message", {}).get("content", "")
                
                # Extract usage
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                # Cost (OpenRouter includes this)
                cost = float(data.get("usage", {}).get("total_cost", 0) or 0)
                
                with self._lock:
                    self.total_cost += cost
                    self.total_calls += 1
                
                return {
                    "response": response_text,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "cost": cost,
                    "latency_ms": latency_ms,
                    "model": model,
                }
                
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.timeout}s"
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                time.sleep(2 ** attempt)
        
        # All retries failed
        return {
            "response": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "cost": 0.0,
            "latency_ms": 0,
            "model": model,
            "error": last_error,
        }
    
    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_cost_usd": round(self.total_cost, 4),
                "total_calls": self.total_calls,
            }

#!/usr/bin/env python3
"""Reusable GPT-4o client for LACL's LLM-based semantic augmentation.

Design goals
------------
* The API key is **never hardcoded**. It is read from the ``OPENAI_API_KEY``
  environment variable (or passed explicitly), and only ever shown masked.
* Adapted to the LACL data format: a service / bundle / mashup is described by
  ``name`` / ``keywords`` (or ``text_description``) / ``category``; the client
  takes a system prompt + a JSON user payload and returns parsed JSON.
* Robust for batch jobs: bounded retries with exponential backoff and tolerant
  JSON extraction (handles ```json fences / extra prose around the object).

Quick start
-----------
    export OPENAI_API_KEY="sk-...your key..."      # key stays in the env only
    python llm_client.py                            # runs the demo below

Use from code
-------------
    from llm_client import LLMClient
    client = LLMClient(model="gpt-4o")
    out = client.chat_json(system_prompt, '{"name": "...", "keywords": "...", "category": "..."}')
"""
from __future__ import annotations

import json
import os
import re
import sys
import time


def mask_key(key: str | None) -> str:
    """Return a desensitized form of an API key for safe logging."""
    if not key:
        return "<unset>"
    if len(key) <= 12:
        return key[:3] + "***"
    return f"{key[:6]}…{key[-4:]} (len={len(key)})"


def extract_json(text: str | None):
    """Best-effort parse of a JSON object from a model response."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start:end + 1])
        except Exception:
            return None
    return None


class LLMClient:
    """Thin wrapper around the OpenAI Chat Completions API (default: gpt-4o)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 5,
        base_url: str | None = None,
    ):
        # Key precedence: explicit arg > environment. Never a literal default.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it first:\n"
                "    export OPENAI_API_KEY=sk-..."
            )
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Imported lazily so the module can be inspected without the dependency.
        from openai import OpenAI

        # base_url lets you point at an OpenAI-compatible endpoint (e.g. another
        # LLM vendor) without changing any calling code — see the README note.
        kwargs = {"api_key": self.api_key}
        if base_url or os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url or os.environ["OPENAI_BASE_URL"]
        self._client = OpenAI(**kwargs)
        print(f"[LLMClient] model={self.model} key={mask_key(self.api_key)}")

    def chat(self, system: str, user: str) -> str:
        """Single chat turn; returns the raw assistant text. Retries on error."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = min(2 ** attempt, 30)
                sys.stderr.write(
                    f"[LLMClient retry {attempt + 1}/{self.max_retries}] {e}; "
                    f"sleep {wait}s\n"
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def chat_json(self, system: str, user: str):
        """Like :meth:`chat` but returns a parsed JSON object (or None)."""
        return extract_json(self.chat(system, user))


def _demo():
    """Minimal end-to-end demo adapted to the LACL service format."""
    from pathlib import Path

    here = Path(__file__).resolve().parent
    system = (here / "service" / "service_description_prompt.txt").read_text(encoding="utf-8")

    # One item in LACL's format: name / keywords / category.
    sample = {
        "name": "google-maps",
        "keywords": "google map service static map street view geocoding "
                    "places directions distance matrix elevation embed javascript",
        "category": "mapping, viewer",
    }
    client = LLMClient(model=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    result = client.chat_json(system, json.dumps(sample, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()

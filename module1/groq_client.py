"""Shared Groq integration with validated JSON responses.

This module keeps API concerns out of prompt construction:
timeouts, retries, rate-limit backoff, JSON extraction, Pydantic validation,
structured logging, and small in-memory caching for deterministic prompts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, TypeVar

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_REASONING_MODEL = os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b")
DEFAULT_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
DEFAULT_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
_CACHE_DISABLED = os.getenv("MEDIAI_DISABLE_LLM_CACHE", "").lower() in {"1", "true", "yes"}

T = TypeVar("T", bound=BaseModel)


class GroqConfigurationError(RuntimeError):
    """Raised when the Groq client cannot be configured safely."""


class GroqJSONError(RuntimeError):
    """Raised when the model cannot produce valid schema-conformant JSON."""


@dataclass(frozen=True)
class GroqJsonResult:
    data: BaseModel
    raw_text: str
    model: str
    tokens_used: int | None
    attempts: int
    cached: bool = False


class _TTLCache:
    def __init__(self, max_items: int = 64, ttl_seconds: int = 900) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, GroqJsonResult]] = OrderedDict()

    def get(self, key: str) -> GroqJsonResult | None:
        if _CACHE_DISABLED:
            return None
        item = self._items.get(key)
        if item is None:
            return None
        created_at, value = item
        if time.time() - created_at > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return GroqJsonResult(
            data=value.data,
            raw_text=value.raw_text,
            model=value.model,
            tokens_used=value.tokens_used,
            attempts=value.attempts,
            cached=True,
        )

    def set(self, key: str, value: GroqJsonResult) -> None:
        if _CACHE_DISABLED:
            return
        self._items[key] = (time.time(), value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


_CACHE = _TTLCache()


def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqConfigurationError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)


def _json_schema_response_format(schema_model: type[BaseModel], schema_name: str, strict: bool) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema_model.model_json_schema(),
            "strict": strict,
        },
    }


def _json_object_response_format() -> dict[str, str]:
    return {"type": "json_object"}


def _with_schema_instruction(messages: list[dict[str, str]], schema_text: str) -> list[dict[str, str]]:
    """Embed the JSON Schema in the prompt.

    Groq's ``json_schema`` response_format rejects Pydantic schemas (strict mode
    requires every property in ``required``; non-strict mode fails generation with
    reasoning models), so we use ``json_object`` mode and describe the schema in
    the prompt instead. This is the combination that validates reliably.
    """
    instruction = (
        "Return exactly ONE JSON object that conforms to this JSON Schema. "
        "Include every required field, use the exact key names, and add no extra keys, "
        "markdown, or commentary.\nJSON Schema:\n" + schema_text
    )
    out = list(messages)
    for idx, msg in enumerate(out):
        if msg.get("role") == "system":
            out[idx] = {**msg, "content": msg["content"].rstrip() + "\n\n" + instruction}
            return out
    out.insert(0, {"role": "system", "content": instruction})
    return out


def _stable_cache_key(
    *,
    model: str,
    messages: Iterable[dict[str, str]],
    schema_name: str,
    temperature: float,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": list(messages),
            "schema_name": schema_name,
            "temperature": temperature,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("no JSON object found", text, 0)


def _usage_total_tokens(completion: Any) -> int | None:
    usage = getattr(completion, "usage", None)
    return getattr(usage, "total_tokens", None) if usage is not None else None


def _is_transient_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 422, 424, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "rate", "connection", "server"))


def _sleep_for_attempt(attempt: int, exc: Exception) -> None:
    retry_after = getattr(exc, "response", None)
    delay = min(8.0, 0.75 * (2 ** max(attempt - 1, 0)))
    try:
        headers = getattr(retry_after, "headers", {}) or {}
        if "retry-after" in headers:
            delay = min(15.0, float(headers["retry-after"]))
    except Exception:
        pass
    time.sleep(delay)


def _repair_messages(
    *,
    raw_text: str,
    validation_error: Exception,
    schema_model: type[BaseModel],
) -> list[dict[str, str]]:
    schema = json.dumps(schema_model.model_json_schema(), ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "Repair the assistant output into one valid JSON object that conforms exactly "
                "to the provided JSON Schema. Do not add markdown or explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                "JSON Schema:\n"
                f"{schema}\n\n"
                "Invalid output:\n"
                f"{raw_text}\n\n"
                "Validation error:\n"
                f"{validation_error}\n\n"
                "Return only corrected JSON."
            ),
        },
    ]


class GroqJsonClient:
    """Small wrapper around Groq chat completions for validated JSON."""

    def call_json(
        self,
        *,
        messages: list[dict[str, str]],
        schema_model: type[T],
        schema_name: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        reasoning_effort: str = "low",
        use_cache: bool = True,
    ) -> GroqJsonResult:
        selected_model = model or DEFAULT_REASONING_MODEL
        cache_key = _stable_cache_key(
            model=selected_model,
            messages=messages,
            schema_name=schema_name,
            temperature=temperature,
        )
        if use_cache:
            cached = _CACHE.get(cache_key)
            if cached is not None:
                logger.info("groq_json_cache_hit", extra={"schema": schema_name, "model": selected_model})
                return cached

        client = _client()
        # Groq's json_schema response_format is unreliable for Pydantic schemas, so we
        # use json_object mode and inject the schema into the prompt (see _with_schema_instruction).
        schema_text = json.dumps(schema_model.model_json_schema(), ensure_ascii=False)
        current_messages = _with_schema_instruction(list(messages), schema_text)
        current_max_tokens = max_tokens
        raw_text = ""
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": selected_model,
                    "messages": current_messages,
                    "temperature": temperature,
                    "max_tokens": current_max_tokens,
                    "response_format": _json_object_response_format(),
                    "timeout": timeout,
                    "seed": 42,
                }
                if selected_model.startswith("openai/gpt-oss"):
                    request["reasoning_effort"] = reasoning_effort
                    request["reasoning_format"] = "hidden"

                completion = client.chat.completions.create(**request)
                raw_text = completion.choices[0].message.content or ""
                parsed = _extract_json_object(raw_text)
                data = schema_model.model_validate(parsed)
                result = GroqJsonResult(
                    data=data,
                    raw_text=json.dumps(data.model_dump(mode="json"), ensure_ascii=False),
                    model=selected_model,
                    tokens_used=_usage_total_tokens(completion),
                    attempts=attempt,
                )
                if use_cache:
                    _CACHE.set(cache_key, result)
                logger.info(
                    "groq_json_success",
                    extra={"schema": schema_name, "model": selected_model, "attempt": attempt},
                )
                return result
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "groq_json_validation_failed",
                    extra={"schema": schema_name, "model": selected_model, "attempt": attempt},
                )
                current_messages = _with_schema_instruction(
                    _repair_messages(raw_text=raw_text, validation_error=exc, schema_model=schema_model),
                    schema_text,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "groq_json_request_failed",
                    extra={
                        "schema": schema_name,
                        "model": selected_model,
                        "attempt": attempt,
                        "error_type": exc.__class__.__name__,
                    },
                )
                # Free-tier per-request token cap: shrink the output budget and retry.
                if getattr(exc, "status_code", None) == 413 and current_max_tokens > 1024:
                    current_max_tokens = max(1024, current_max_tokens // 2)
                    continue
                if attempt < max_retries and _is_transient_error(exc):
                    _sleep_for_attempt(attempt, exc)
                    continue
                if attempt >= max_retries:
                    break

        raise GroqJSONError(f"Groq JSON response failed validation: {last_error}")

"""Small OpenAI-compatible tool-calling client. No implicit offline fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse


class ProviderError(RuntimeError):
    """A sanitized failure suitable for a user interface."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError(
            "LLM endpoint redirected. Configure the final HTTPS endpoint explicitly."
        )


@dataclass
class OpenAIPlanner:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0

    @classmethod
    def from_env(cls):
        key = os.environ.get("NARRACARE_LLM_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        if not key:
            raise ProviderError(
                "LLM not connected. Set OPENAI_API_KEY securely in the environment. No offline agent is substituted."
            )
        return cls(
            key,
            os.environ.get("NARRACARE_LLM_MODEL", "gpt-4o-mini"),
            os.environ.get("NARRACARE_LLM_BASE_URL", "https://api.openai.com/v1"),
        )

    def complete(self, messages: list, tools: list) -> dict:
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ProviderError(
                "LLM base URL must be HTTPS without embedded credentials."
            )
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": 0,
            "max_tokens": 800,
        }
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.build_opener(NoRedirect).open(
                req, timeout=self.timeout
            ) as response:
                data = json.loads(response.read(2_000_000))
            message = data["choices"][0]["message"]
            if (
                not isinstance(message.get("tool_calls"), list)
                or not message["tool_calls"]
            ):
                raise ProviderError(
                    "The model returned no tool call. Use a tool-calling capable model."
                )
            # Persist only the fields required by the conversation protocol.
            return {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message["tool_calls"],
            }
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"LLM request failed (HTTP {exc.code}). Check model, endpoint and credentials."
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ProviderError(
                "LLM connection failed or timed out. No recommendation was fabricated."
            ) from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError(
                "LLM returned an invalid tool-calling response."
            ) from None

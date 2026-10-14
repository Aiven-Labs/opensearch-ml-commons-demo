#!/usr/bin/env python3
"""Create an Aiven AI Gateway key and store it without printing the secret."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
PROJECT = "dimakan-search-playground"
KEY_NAME = "search-demo"
EXPIRY_TIME = "2026-10-19T00:00:00Z"
URL = f"https://api.aiven.io/v1/project/{PROJECT}/ai-gateway/keys"
SECRET_FIELD_NAMES = {"key", "token", "secret", "api_key", "full_token", "value"}


def load_env() -> list[str]:
    lines = ENV_PATH.read_text().splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))
    return lines


def find_secret(value: Any, path: tuple[str, ...] = ()) -> tuple[str, tuple[str, ...]]:
    candidates: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = (*path, key)
            if key.lower() in SECRET_FIELD_NAMES and isinstance(item, str) and item:
                candidates.append((item, item_path))
            if isinstance(item, (dict, list)):
                try:
                    candidates.append(find_secret(item, item_path))
                except ValueError:
                    pass
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                try:
                    candidates.append(find_secret(item, (*path, str(index))))
                except ValueError:
                    pass
    if not candidates:
        raise ValueError("No secret field found in the API response")
    return max(candidates, key=lambda candidate: len(candidate[0]))


def update_env(lines: list[str], name: str, value: str) -> None:
    prefix = f"{name}="
    updated = False
    output: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            output.append(f"{prefix}{value}")
            updated = True
        else:
            output.append(line)
    if not updated:
        output.append(f"{prefix}{value}")
    ENV_PATH.write_text("\n".join(output) + "\n")


def main() -> None:
    lines = load_env()
    token = os.getenv("AIVEN_TOKEN", "")
    if not token:
        raise SystemExit("AIVEN_TOKEN is missing from .env")

    request = urllib.request.Request(
        URL,
        data=json.dumps({"name": KEY_NAME, "expiry_time": EXPIRY_TIME}).encode(),
        method="POST",
        headers={
            "Authorization": f"aivenv1 {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise SystemExit(f"Aiven returned HTTP {error.code}: {detail}") from error

    try:
        secret, field_path = find_secret(result)
    except ValueError as error:
        safe_keys = list(result) if isinstance(result, dict) else []
        raise SystemExit(
            f"Key was created, but its secret field was not recognized. "
            f"Top-level response fields: {safe_keys}"
        ) from error

    update_env(lines, "LLM_API_KEY", secret)
    print(
        json.dumps(
            {
                "created": True,
                "project": PROJECT,
                "name": KEY_NAME,
                "expiry_time": EXPIRY_TIME,
                "stored_in": ".env:LLM_API_KEY",
                "response_field": ".".join(field_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect and provision an OpenSearch-provided local embedding model."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "local-models.json"
TERMINAL_STATES = {"COMPLETED", "FAILED", "COMPLETED_WITH_ERROR"}


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not os.environ.get(key):
            os.environ[key] = value.strip().strip("'\"")


def catalog() -> list[dict[str, Any]]:
    return json.loads(CATALOG_PATH.read_text())


def selected_model() -> dict[str, Any]:
    name = os.getenv("LOCAL_MODEL_NAME")
    if not name:
        raise SystemExit("Set LOCAL_MODEL_NAME in .env (see .env.example).")
    for model in catalog():
        if model["name"] == name:
            return model
    choices = "\n  ".join(model["name"] for model in catalog())
    raise SystemExit(f"Unknown LOCAL_MODEL_NAME={name!r}. Available names:\n  {choices}")


def connection() -> tuple[str, str]:
    required = ("OPENSEARCH_URL", "OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")
    url = os.environ["OPENSEARCH_URL"].rstrip("/")
    token = base64.b64encode(
        f"{os.environ['OPENSEARCH_USERNAME']}:{os.environ['OPENSEARCH_PASSWORD']}".encode()
    ).decode()
    return url, f"Basic {token}"


def request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    url, authorization = connection()
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{url}{path}",
        data=payload,
        method=method,
        headers={"Authorization": authorization, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, context=ssl.create_default_context()) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise SystemExit(f"OpenSearch returned HTTP {error.code}: {detail}") from error


def cluster_models() -> list[dict[str, Any]]:
    result = request(
        "POST",
        "/_plugins/_ml/models/_search",
        {"size": 100, "query": {"match_all": {}}},
    )
    return [
        {
            "model_id": hit["_id"],
            "name": hit["_source"].get("name"),
            "version": hit["_source"].get("model_version"),
            "format": hit["_source"].get("model_format"),
            "function": hit["_source"].get("function_name"),
            "state": hit["_source"].get("model_state"),
        }
        for hit in result["hits"]["hits"]
        if hit["_source"].get("model_state") is not None
    ]


def wait_for_task(task_id: str) -> dict[str, Any]:
    while True:
        task = request("GET", f"/_plugins/_ml/tasks/{task_id}")
        state = task.get("state")
        print(f"Task {task_id}: {state}", file=sys.stderr)
        if state in TERMINAL_STATES:
            if state != "COMPLETED":
                raise SystemExit(json.dumps(task, indent=2))
            return task
        time.sleep(2)


def register(model: dict[str, Any]) -> str:
    existing = [
        item
        for item in cluster_models()
        if item["name"] == model["name"]
    ]
    if existing:
        model_id = existing[0]["model_id"]
        print(f"Already registered: {model_id}", file=sys.stderr)
        return model_id

    result = request(
        "POST",
        "/_plugins/_ml/models/_register",
        {
            "name": model["name"],
            "version": model["version"],
            "model_format": model["format"],
        },
    )
    task = wait_for_task(result["task_id"])
    model_id = task.get("model_id")
    if not model_id:
        raise SystemExit(f"Registration completed without a model_id:\n{json.dumps(task, indent=2)}")
    return model_id


def deploy(model_id: str) -> None:
    result = request("POST", f"/_plugins/_ml/models/{model_id}/_deploy")
    wait_for_task(result["task_id"])


def test_embedding(model: dict[str, Any], text: str) -> None:
    existing = [
        item
        for item in cluster_models()
        if item["name"] == model["name"]
    ]
    if not existing:
        raise SystemExit("The selected model is not registered. Run the provision command first.")

    model_id = existing[0]["model_id"]
    result = request(
        "POST",
        f"/_plugins/_ml/_predict/text_embedding/{model_id}",
        {"text_docs": [text]},
    )
    vectors = [
        output["data"]
        for inference in result.get("inference_results", [])
        for output in inference.get("output", [])
        if output.get("name") == "sentence_embedding"
        and isinstance(output.get("data"), list)
    ]
    if not vectors:
        raise SystemExit(f"No embedding found in response:\n{json.dumps(result, indent=2)}")

    vector = vectors[0]
    observed_dimension = len(vector)
    expected_dimension = model["dimension"]
    print(
        json.dumps(
            {
                "model_id": model_id,
                "input": text,
                "expected_dimension": expected_dimension,
                "observed_dimension": observed_dimension,
                "dimension_matches": observed_dimension == expected_dimension,
                "l2_norm": round(math.sqrt(sum(value * value for value in vector)), 6),
                "first_8_values": vector[:8],
            },
            indent=2,
        )
    )
    if observed_dimension != expected_dimension:
        raise SystemExit("Embedding dimension does not match the model catalog.")


def print_catalog() -> None:
    for model in catalog():
        print(
            f"{model['name']}\n"
            f"  version={model['version']} dimension={model['dimension']} "
            f"format={model['format']}\n"
            f"  {model['purpose']}"
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("catalog", "selected", "cluster", "register", "provision", "test"),
    )
    parser.add_argument(
        "--text",
        default="How does OpenSearch run machine learning models?",
        help="Input text for the test command.",
    )
    args = parser.parse_args()

    if args.command == "catalog":
        print_catalog()
        return
    if args.command == "cluster":
        print(json.dumps(cluster_models(), indent=2))
        return

    model = selected_model()
    if args.command == "selected":
        print(json.dumps(model, indent=2))
        return
    if args.command == "test":
        test_embedding(model, args.text)
        return

    model_id = register(model)
    if args.command == "provision":
        deploy(model_id)
    print(
        json.dumps(
            {
                "model_id": model_id,
                "name": model["name"],
                "dimension": model["dimension"],
                "deployed": args.command == "provision",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

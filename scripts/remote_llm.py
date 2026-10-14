#!/usr/bin/env python3
"""Provision and test an OpenAI-compatible LLM connector."""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from typing import Any

from local_model import load_dotenv, request, wait_for_task


def setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Set {name} in .env (see .env.example).")
    return value


def configuration() -> dict[str, str]:
    base_url = os.getenv("AIVEN_AI_API_BASE", "").strip() or setting("LLM_BASE_URL")
    base_url = base_url.rstrip("/")
    if not base_url.startswith("https://"):
        raise SystemExit(
            "The LLM API base must be an HTTPS URL reachable from the Aiven cluster."
        )
    model = os.getenv("AIVEN_AI_MODEL", "").strip() or setting("LLM_MODEL_NAME")
    api_key = os.getenv("AIVEN_AI_API_KEY", "").strip() or setting("LLM_API_KEY")
    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "connector_name": os.getenv(
            "LLM_CONNECTOR_NAME", "openai-compatible-llm"
        ),
        "remote_model_name": os.getenv(
            "LLM_REMOTE_MODEL_NAME", "openai-compatible-chat"
        ),
    }


def allowlist_regex(base_url: str) -> str:
    return f"^{re.escape(base_url)}/.*$"


def local_endpoint_test(config: dict[str, str], prompt: str) -> str:
    body = json.dumps(
        {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 128,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{config['base_url']}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, context=ssl.create_default_context()) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise SystemExit(f"LLM endpoint returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Cannot reach LLM endpoint: {error.reason}") from error

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise SystemExit(
            f"Endpoint is not OpenAI chat compatible:\n{json.dumps(result, indent=2)}"
        ) from error


def connectors() -> list[dict[str, Any]]:
    result = request(
        "POST",
        "/_plugins/_ml/connectors/_search",
        {"size": 100, "query": {"match_all": {}}},
    )
    return [
        {
            "connector_id": hit["_id"],
            "name": hit["_source"].get("name"),
            "description": hit["_source"].get("description"),
        }
        for hit in result["hits"]["hits"]
    ]


def create_connector(config: dict[str, str]) -> str:
    existing = [
        connector
        for connector in connectors()
        if connector["name"] == config["connector_name"]
    ]
    if existing:
        connector_id = existing[0]["connector_id"]
        print(f"Using existing connector: {connector_id}", file=sys.stderr)
        return connector_id

    result = request(
        "POST",
        "/_plugins/_ml/connectors/_create",
        {
            "name": config["connector_name"],
            "description": (
                "OpenAI-compatible connector to Aiven AI Gateway"
            ),
            "version": 1,
            "protocol": "http",
            "parameters": {
                "endpoint": config["base_url"],
                "model": config["model"],
                "temperature": 0,
                "max_tokens": 256,
            },
            "credential": {"api_key": config["api_key"]},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "${parameters.endpoint}/chat/completions",
                    "headers": {
                        "Authorization": "Bearer ${credential.api_key}",
                        "Content-Type": "application/json",
                    },
                    "request_body": (
                        '{"model":"${parameters.model}",'
                        '"messages":[{"role":"user",'
                        '"content":"${parameters.prompt}"}],'
                        '"temperature":${parameters.temperature},'
                        '"max_tokens":${parameters.max_tokens},'
                        '"stream":false}'
                    ),
                }
            ],
            "client_config": {
                "connection_timeout": 10,
                "read_timeout": 60,
                "max_retry_times": 1,
            },
        },
    )
    return result["connector_id"]


def remote_models() -> list[dict[str, Any]]:
    result = request(
        "POST",
        "/_plugins/_ml/models/_search",
        {"size": 100, "query": {"match_all": {}}},
    )
    return [
        {
            "model_id": hit["_id"],
            "name": hit["_source"].get("name"),
            "state": hit["_source"].get("model_state"),
        }
        for hit in result["hits"]["hits"]
        if hit["_source"].get("model_state") is not None
    ]


def register_model(config: dict[str, str], connector_id: str) -> str:
    existing = [
        model
        for model in remote_models()
        if model["name"] == config["remote_model_name"]
    ]
    if existing:
        model_id = existing[0]["model_id"]
        print(f"Using existing remote model: {model_id}", file=sys.stderr)
        return model_id

    result = request(
        "POST",
        "/_plugins/_ml/models/_register",
        {
            "name": config["remote_model_name"],
            "function_name": "remote",
            "description": (
                f"{config['model']} through an OpenAI-compatible API"
            ),
            "connector_id": connector_id,
        },
    )
    if "model_id" in result:
        return result["model_id"]
    task = wait_for_task(result["task_id"])
    if "model_id" not in task:
        raise SystemExit(f"Registration returned no model_id:\n{json.dumps(task, indent=2)}")
    return task["model_id"]


def deploy_model(model_id: str) -> None:
    model = request("GET", f"/_plugins/_ml/models/{model_id}")
    if model.get("model_state") == "DEPLOYED":
        print(f"Model is already deployed: {model_id}", file=sys.stderr)
        return
    result = request("POST", f"/_plugins/_ml/models/{model_id}/_deploy")
    wait_for_task(result["task_id"])


def find_model_id(config: dict[str, str]) -> str:
    matches = [
        model
        for model in remote_models()
        if model["name"] == config["remote_model_name"]
    ]
    if not matches:
        raise SystemExit("Remote model is not registered. Run the provision command first.")
    return matches[0]["model_id"]


def predict(model_id: str, prompt: str) -> dict[str, Any]:
    return request(
        "POST",
        f"/_plugins/_ml/models/{model_id}/_predict",
        {
            "parameters": {
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 128,
            }
        },
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "config",
            "allowlist",
            "endpoint-test",
            "connectors",
            "provision",
            "test",
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Explain hybrid search in one concise sentence.",
    )
    args = parser.parse_args()
    config = configuration()

    if args.command == "config":
        print(
            json.dumps(
                {
                    "base_url": config["base_url"],
                    "model": config["model"],
                    "connector_name": config["connector_name"],
                    "remote_model_name": config["remote_model_name"],
                    "api_key_configured": True,
                },
                indent=2,
            )
        )
        return
    if args.command == "allowlist":
        print(
            json.dumps(
                {
                    "aiven_advanced_configuration": {
                        "opensearch.ml_commons_trusted_connector_endpoints_regex": [
                            allowlist_regex(config["base_url"])
                        ]
                    }
                },
                indent=2,
            )
        )
        return
    if args.command == "endpoint-test":
        print(local_endpoint_test(config, args.prompt))
        return
    if args.command == "connectors":
        print(json.dumps(connectors(), indent=2))
        return
    if args.command == "provision":
        connector_id = create_connector(config)
        model_id = register_model(config, connector_id)
        deploy_model(model_id)
        print(json.dumps({"connector_id": connector_id, "model_id": model_id}, indent=2))
        return

    model_id = find_model_id(config)
    print(json.dumps(predict(model_id, args.prompt), indent=2))


if __name__ == "__main__":
    main()

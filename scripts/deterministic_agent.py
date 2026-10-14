#!/usr/bin/env python3
"""Provision and run the deterministic conversational RAG demo agent."""

from __future__ import annotations

import argparse
import json
from typing import Any

from local_model import cluster_models, load_dotenv, request, selected_model


AGENT_NAME = "opensearch-docs-deterministic-agent"
INDEX = "talk-docs"
EMBEDDING_FIELD = "embedding"
REMOTE_MODEL_NAME = "aiven-claude-sonnet-5"


def model_id(name: str) -> str:
    matches = [
        model
        for model in cluster_models()
        if model["name"] == name and model["state"] == "DEPLOYED"
    ]
    if not matches:
        raise SystemExit(f"Deployed model not found: {name}")
    return matches[0]["model_id"]


def agents() -> list[dict[str, Any]]:
    indices = request(
        "GET", "/_cat/indices?format=json&expand_wildcards=all&h=index"
    )
    if not any(item.get("index") == ".plugins-ml-agent" for item in indices):
        return []
    result = request(
        "POST",
        "/_plugins/_ml/agents/_search",
        {"size": 100, "query": {"match_all": {}}},
    )
    return [
        {
            "agent_id": hit["_id"],
            "name": hit["_source"].get("name"),
            "type": hit["_source"].get("type"),
        }
        for hit in result["hits"]["hits"]
    ]


def find_agent() -> str | None:
    matches = [agent for agent in agents() if agent["name"] == AGENT_NAME]
    return matches[0]["agent_id"] if matches else None


def provision() -> str:
    existing = find_agent()
    if existing:
        print(f"Using existing agent: {existing}")
        return existing

    embedding_model_id = model_id(selected_model()["name"])
    llm_model_id = model_id(REMOTE_MODEL_NAME)
    result = request(
        "POST",
        "/_plugins/_ml/agents/_register",
        {
            "name": AGENT_NAME,
            "type": "conversational_flow",
            "description": (
                "Fixed retrieval-then-generation agent for the OpenSearch AI talk"
            ),
            "app_type": "rag",
            "memory": {"type": "conversation_index"},
            "tools": [
                {
                    "type": "VectorDBTool",
                    "name": "knowledge_base",
                    "description": "Retrieve relevant OpenSearch documentation",
                    "parameters": {
                        "model_id": embedding_model_id,
                        "index": INDEX,
                        "embedding_field": EMBEDDING_FIELD,
                        "source_field": ["title", "text", "url"],
                        "doc_size": 3,
                        "k": 10,
                        "input": "${parameters.question}",
                    },
                },
                {
                    "type": "MLModelTool",
                    "name": "grounded_answer",
                    "description": "Answer only from the retrieved documentation",
                    "parameters": {
                        "model_id": llm_model_id,
                        "prompt": (
                            "You are an OpenSearch documentation assistant. "
                            "Answer only from the retrieved context or the "
                            "conversation history below. "
                            "For a follow-up that asks to summarize, shorten, "
                            "rephrase, or otherwise transform a previous answer, "
                            "use the most recent assistant answer in conversation "
                            "history and do not require the newly retrieved context "
                            "to repeat it. "
                            "Cite the source URLs used. If the context does not "
                            "support the answer, say exactly: I don't know based "
                            "on the indexed documentation. Keep the answer under "
                            "120 words.\n\n"
                            "Retrieved context:\n"
                            "${parameters.knowledge_base.output:-}\n\n"
                            "Conversation history:\n"
                            "${parameters.chat_history:-}\n\n"
                            "Question: ${parameters.question}\n"
                            "Answer:"
                        ),
                        "temperature": 0,
                        "max_tokens": 400,
                    },
                },
            ],
        },
    )
    return result["agent_id"]


def execute(agent_id: str, question: str, memory_id: str | None = None) -> dict[str, Any]:
    parameters: dict[str, Any] = {"question": question, "verbose": True}
    if memory_id:
        parameters["memory_id"] = memory_id
    return request(
        "POST",
        f"/_plugins/_ml/agents/{agent_id}/_execute",
        {"parameters": parameters},
    )


def outputs(result: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for inference in result.get("inference_results", []):
        for output in inference.get("output", []):
            name = output.get("name", "response")
            if "result" in output:
                values[name] = output["result"]
            elif "dataAsMap" in output:
                values[name] = output["dataAsMap"]
    return values


def answer_payload(result: dict[str, Any]) -> dict[str, Any]:
    parsed = outputs(result)
    response = parsed.get("response") or parsed.get("grounded_answer")
    provider_response: dict[str, Any] | None = None
    if isinstance(response, str):
        try:
            decoded = json.loads(response)
            if isinstance(decoded, dict):
                provider_response = decoded
        except json.JSONDecodeError:
            pass

    display: dict[str, Any] = {"memory_id": parsed.get("memory_id")}
    if provider_response:
        try:
            display["answer"] = provider_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            display["answer"] = provider_response
        display["model"] = provider_response.get("model")
        display["token_usage"] = provider_response.get("usage")
    else:
        display["answer"] = response
    return display


def show_answer(result: dict[str, Any]) -> str | None:
    display = answer_payload(result)
    print(json.dumps(display, indent=2))
    memory_id = display.get("memory_id")
    return memory_id if isinstance(memory_id, str) else None


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("provision", "rebuild", "ask", "demo", "list")
    )
    parser.add_argument(
        "--question",
        default="Why should local models run on dedicated ML nodes?",
    )
    parser.add_argument("--memory-id")
    args = parser.parse_args()

    if args.command == "list":
        print(json.dumps(agents(), indent=2))
        return
    if args.command == "rebuild":
        existing = find_agent()
        if existing:
            request("DELETE", f"/_plugins/_ml/agents/{existing}")

    agent_id = provision()
    if args.command in {"provision", "rebuild"}:
        print(json.dumps({"agent_id": agent_id, "name": AGENT_NAME}, indent=2))
        return

    first = execute(agent_id, args.question, args.memory_id)
    memory_id = show_answer(first)
    if args.command == "demo":
        if not memory_id:
            raise SystemExit("The first response did not return a memory_id.")
        print("\n--- follow-up using the same memory_id ---\n")
        follow_up = execute(
            agent_id,
            "What default limits and circuit breakers should I watch?",
            memory_id,
        )
        show_answer(follow_up)


if __name__ == "__main__":
    main()

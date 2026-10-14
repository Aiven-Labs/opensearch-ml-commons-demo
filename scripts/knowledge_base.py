#!/usr/bin/env python3
"""Build and query the documentation knowledge base used by the demo agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from local_model import cluster_models, load_dotenv, request, selected_model


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "opensearch-docs.json"
INDEX = "talk-docs"
PIPELINE = "talk-embedding"
HYBRID_PIPELINE = "talk-rrf"


def embedding_model() -> tuple[dict[str, Any], str]:
    selected = selected_model()
    matches = [
        model
        for model in cluster_models()
        if model["name"] == selected["name"] and model["state"] == "DEPLOYED"
    ]
    if not matches:
        raise SystemExit("The selected local embedding model is not deployed.")
    return selected, matches[0]["model_id"]


def index_exists() -> bool:
    result = request("GET", "/_cat/indices?format=json&h=index")
    return any(item.get("index") == INDEX for item in result)


def create_pipeline(model_id: str) -> None:
    request(
        "PUT",
        f"/_ingest/pipeline/{PIPELINE}",
        {
            "description": "Generate local embeddings for the talk knowledge base",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": model_id,
                        "field_map": {"text": "embedding"},
                    }
                }
            ],
        },
    )


def create_hybrid_pipeline() -> None:
    request(
        "PUT",
        f"/_search/pipeline/{HYBRID_PIPELINE}",
        {
            "description": "Combine lexical and neural ranks using RRF",
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf",
                            "rank_constant": 60,
                        }
                    }
                }
            ],
        },
    )


def create_index(dimension: int) -> None:
    request(
        "PUT",
        f"/{INDEX}",
        {
            "settings": {
                "index": {
                    "knn": True,
                    "default_pipeline": PIPELINE,
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                    "url": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                        },
                    },
                }
            },
        },
    )


def setup(rebuild: bool) -> None:
    model, model_id = embedding_model()
    create_pipeline(model_id)
    create_hybrid_pipeline()
    exists = index_exists()
    if rebuild and exists:
        request("DELETE", f"/{INDEX}")
        exists = False
    if not exists:
        create_index(model["dimension"])

    documents = json.loads(CORPUS_PATH.read_text())
    for document in documents:
        document_id = quote(document["id"], safe="")
        request("PUT", f"/{INDEX}/_doc/{document_id}", document)
    request("POST", f"/{INDEX}/_refresh")
    count = request("GET", f"/{INDEX}/_count")["count"]
    print(
        json.dumps(
            {
                "index": INDEX,
                "pipeline": PIPELINE,
                "embedding_model_id": model_id,
                "dimension": model["dimension"],
                "documents": count,
            },
            indent=2,
        )
    )


def search_results(query: str, size: int) -> dict[str, list[dict[str, Any]]]:
    _, model_id = embedding_model()
    source = {"excludes": ["embedding"]}
    keyword = request(
        "POST",
        f"/{INDEX}/_search",
        {
            "_source": source,
            "size": size,
            "query": {"match": {"text": query}},
        },
    )
    vector = request(
        "POST",
        f"/{INDEX}/_search",
        {
            "_source": source,
            "size": size,
            "query": {
                "neural": {
                    "embedding": {
                        "query_text": query,
                        "model_id": model_id,
                        "k": max(size, 10),
                    }
                }
            },
        },
    )
    hybrid = request(
        "POST",
        f"/{INDEX}/_search?search_pipeline={HYBRID_PIPELINE}",
        {
            "_source": source,
            "size": size,
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"text": query}},
                        {
                            "neural": {
                                "embedding": {
                                    "query_text": query,
                                    "model_id": model_id,
                                    "k": max(size, 10),
                                }
                            }
                        },
                    ]
                }
            },
        },
    )

    def hits(result: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "rank": rank,
                "score": hit["_score"],
                "id": hit["_id"],
                **hit["_source"],
            }
            for rank, hit in enumerate(result["hits"]["hits"], start=1)
        ]

    return {
        "keyword": hits(keyword),
        "vector": hits(vector),
        "hybrid": hits(hybrid),
    }


def neural_search(query: str, size: int) -> None:
    result = search_results(query, size)["vector"]
    print(
        json.dumps(result, indent=2)
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("setup", "rebuild", "search", "compare", "status")
    )
    parser.add_argument(
        "--query",
        default="Why should local models run on dedicated ML nodes?",
    )
    parser.add_argument("--size", type=int, default=3)
    args = parser.parse_args()

    if args.command in {"setup", "rebuild"}:
        setup(rebuild=args.command == "rebuild")
    elif args.command == "search":
        neural_search(args.query, args.size)
    elif args.command == "compare":
        print(json.dumps(search_results(args.query, args.size), indent=2))
    else:
        if not index_exists():
            raise SystemExit(f"Index {INDEX!r} does not exist.")
        print(json.dumps(request("GET", f"/{INDEX}/_count"), indent=2))


if __name__ == "__main__":
    main()

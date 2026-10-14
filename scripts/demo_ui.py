#!/usr/bin/env python3
"""Serve the localhost-only UI for the deterministic agent demo."""

from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from deterministic_agent import answer_payload, execute, find_agent, provision
from knowledge_base import INDEX, index_exists, search_results
from local_model import load_dotenv, request


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "ui"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_QUESTION_LENGTH = 2_000


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "OpenSearchAgentDemo/1.0"

    def security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'none'",
        )

    def send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.security_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: HTTPStatus, value: Any) -> None:
        self.send_bytes(
            status,
            json.dumps(value).encode(),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ASSETS:
            filename, content_type = ASSETS[path]
            self.send_bytes(HTTPStatus.OK, (UI_ROOT / filename).read_bytes(), content_type)
            return
        if path == "/api/status":
            try:
                count = request("GET", f"/{INDEX}/_count")["count"] if index_exists() else 0
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "agent_id": find_agent(),
                        "agent_type": "conversational_flow",
                        "documents": count,
                        "index": INDEX,
                        "retrieval_tool": "VectorDBTool",
                        "generation_tool": "MLModelTool",
                    },
                )
            except (Exception, SystemExit) as error:
                self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/ask":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            question = str(body.get("question", "")).strip()
            raw_memory_id = body.get("memory_id")
            memory_id = (
                raw_memory_id.strip()
                if isinstance(raw_memory_id, str) and raw_memory_id.strip()
                else None
            )
            include_retrieval_comparison = bool(
                body.get("include_retrieval_comparison", memory_id is None)
            )
            if not question:
                raise ValueError("Question is required")
            if len(question) > MAX_QUESTION_LENGTH:
                raise ValueError(
                    f"Question exceeds {MAX_QUESTION_LENGTH} characters"
                )

            retrieval: dict[str, Any] | None = None
            retrieval_ms = 0
            if include_retrieval_comparison:
                retrieval_started = time.perf_counter()
                retrieval = search_results(question, 3)
                retrieval_ms = round(
                    (time.perf_counter() - retrieval_started) * 1_000
                )

            agent_id = find_agent() or provision()
            agent_started = time.perf_counter()
            result = execute(agent_id, question, memory_id)
            agent_ms = round((time.perf_counter() - agent_started) * 1_000)
            agent_payload = answer_payload(result)
            self.send_json(
                HTTPStatus.OK,
                {
                    "question": question,
                    "retrieval": retrieval,
                    "agent": agent_payload,
                    "agent_id": agent_id,
                    "timings_ms": {
                        "retrieval_comparison": retrieval_ms,
                        "agent_execution": agent_ms,
                        "total": retrieval_ms + agent_ms,
                    },
                    "actual_agent_path": [
                        "question",
                        "VectorDBTool (local embedding + vector retrieval)",
                        "MLModelTool (Claude Sonnet 5 via Aiven AI Gateway)",
                        "conversation memory",
                    ],
                },
            )
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except (Exception, SystemExit) as error:
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Agent demo UI: http://{args.host}:{args.port}")
    print("Credentials remain server-side. Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

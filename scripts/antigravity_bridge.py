#!/usr/bin/env python3
"""Antigravity / agy OpenAI-Compatible API Bridge Server.

Acts as a local OpenAI REST API server (e.g. http://127.0.0.1:8000/v1) that translates
standard /v1/chat/completions requests into local CLI execution (antigravity / agy).

Features:
- Auto-detects whether 'antigravity' or 'agy' CLI binary is installed in PATH.
- Supports /v1/chat/completions (JSON & Server-Sent Events streaming).
- Supports /v1/models and /health endpoints.
- Requires no external pip dependencies (built on Python standard library).

Usage:
  python3 scripts/antigravity_bridge.py [--port 8000] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("antigravity_bridge")


def detect_cli_command() -> Tuple[str, str]:
    """Auto-detect available local CLI binary and command template.

    Returns:
        (cli_binary_name, command_template)
    """
    env_cmd = os.environ.get("ANTIGRAVITY_BRIDGE_CMD", "").strip()
    if env_cmd:
        binary = env_cmd.split()[0]
        return (binary, env_cmd)

    agy_path = shutil.which("agy")
    if agy_path:
        return ("agy", 'agy -p "{prompt}"')

    anti_path = shutil.which("antigravity")
    if anti_path:
        return ("antigravity", 'antigravity -p "{prompt}"')

    # Fallback to agy template if neither is found in PATH at startup
    return ("agy", 'agy -p "{prompt}"')


def format_messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """Format OpenAI messages list into a prompt string for CLI tools."""
    if not messages:
        return ""

    if len(messages) == 1 and messages[0].get("role") == "user":
        content = messages[0].get("content") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )

    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        elif not isinstance(content, str):
            content = str(content)

        if role == "system":
            parts.append(f"[System Instructions]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            tc_text = f"\nTool Calls: {json.dumps(tool_calls)}" if tool_calls else ""
            parts.append(f"[Assistant]\n{content}{tc_text}")
        elif role == "tool":
            parts.append(f"[Tool Result]\n{content}")
        else:
            parts.append(f"[{role.capitalize()}]\n{content}")

    return "\n\n".join(parts)


def execute_cli_command(
    cmd_template: str,
    prompt_text: str,
    timeout: float = 180.0,
) -> str:
    """Execute local CLI command with prompt substitution or stdin piping."""
    if "{prompt}" in cmd_template:
        cmd_str = cmd_template.replace("{prompt}", prompt_text)
        stdin_input = ""
    else:
        cmd_str = cmd_template
        stdin_input = prompt_text

    logger.info("Executing CLI command: %s", cmd_str[:120])

    proc = subprocess.Popen(
        cmd_str,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout_data, stderr_data = proc.communicate(input=stdin_input, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_data, stderr_data = proc.communicate()
        raise RuntimeError("CLI Execution Timeout")

    if proc.returncode != 0:
        err_msg = stderr_data.strip() or stdout_data.strip() or f"Exit code {proc.returncode}"
        logger.error("CLI execution failed (code %d): %s", proc.returncode, err_msg)
        raise RuntimeError(f"CLI Execution Error: {err_msg}")

    return stdout_data.strip()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server for handling concurrent API calls."""
    daemon_threads = True


class AntigravityBridgeHandler(BaseHTTPRequestHandler):
    """HTTP Handler implementing OpenAI ChatCompletions REST API."""

    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._send_cors_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")

        if path in ("", "/health"):
            cli_bin, cmd_tpl = detect_cli_command()
            self._send_json_response({
                "status": "ok",
                "service": "antigravity-bridge",
                "cli_detected": cli_bin,
                "command_template": cmd_tpl,
            })
            return

        if path in ("/v1/models", "/models"):
            cli_bin, _ = detect_cli_command()
            self._send_json_response({
                "object": "list",
                "data": [
                    {
                        "id": "antigravity",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    },
                    {
                        "id": "agy",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    },
                    {
                        "id": cli_bin,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    },
                ],
            })
            return

        self._send_json_response({"error": "Not Found"}, status_code=404)

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")

        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json_response({"error": "Not Found"}, status_code=404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body_data = self.rfile.read(content_length)

        try:
            req_json = json.loads(body_data.decode("utf-8"))
        except Exception as exc:
            self._send_json_response(
                {"error": {"message": f"Invalid JSON payload: {exc}", "type": "invalid_request_error"}},
                status_code=400,
            )
            return

        messages = req_json.get("messages", [])
        model = req_json.get("model") or "antigravity"
        stream = req_json.get("stream", False)

        prompt_text = format_messages_to_prompt(messages)
        cli_bin, cmd_tpl = detect_cli_command()

        # Allow per-request custom template override if passed in extra args
        custom_tpl = getattr(self.server, "custom_cmd", None) or cmd_tpl

        try:
            output_text = execute_cli_command(custom_tpl, prompt_text)
        except Exception as exc:
            logger.error("Chat completion error: %s", exc)
            self._send_json_response(
                {"error": {"message": str(exc), "type": "api_error"}},
                status_code=500,
            )
            return

        completion_id = f"chatcmpl-ag-{uuid.uuid4().hex[:8]}"
        created_ts = int(time.time())

        if stream:
            # Send Server-Sent Events (SSE) stream
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            chunk_start = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": output_text},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk_start)}\n\n".encode("utf-8"))
            self.wfile.flush()

            chunk_stop = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk_stop)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        # Standard non-streaming JSON response
        response_payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_text) // 4,
                "completion_tokens": len(output_text) // 4,
                "total_tokens": (len(prompt_text) + len(output_text)) // 4,
            },
        }

        self._send_json_response(response_payload)


def main():
    parser = argparse.ArgumentParser(
        description="Antigravity / agy OpenAI-compatible API Bridge Server"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--cmd", default=None, help="Custom CLI command template (e.g. 'agy -p \"{prompt}\"')")

    args = parser.parse_args()

    cli_bin, cmd_tpl = detect_cli_command()
    effective_cmd = args.cmd or cmd_tpl

    logger.info("Starting Antigravity API Bridge Server...")
    logger.info("Detected CLI Binary: %s", cli_bin)
    logger.info("Command Template:   %s", effective_cmd)
    logger.info("Listening on:       http://%s:%d/v1", args.host, args.port)

    server = ThreadedHTTPServer((args.host, args.port), AntigravityBridgeHandler)
    server.custom_cmd = effective_cmd

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()

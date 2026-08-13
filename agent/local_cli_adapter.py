"""Local CLI Model Client Adapter for Hermes Agent.

Allows executing LLM completion calls via local CLI tools such as:
- agy -p "{prompt}"
- antigravity chat "{prompt}"
- custom command template or stdin piping
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _format_messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """Format a list of chat messages into a clean text prompt for CLI tools."""
    if not messages:
        return ""

    # Single user message optimization
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


def get_default_cli_command(model: Optional[str] = None) -> str:
    """Determine default CLI command from env, config, or model name."""
    # 1. Environment variable
    env_cmd = os.environ.get("HERMES_LOCAL_CLI_COMMAND")
    if env_cmd and env_cmd.strip():
        return env_cmd.strip()

    # 2. Config YAML
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        model_cfg = cfg.get("model", {})
        cli_cmd = model_cfg.get("cli_command") or model_cfg.get("local_cli_command")
        if cli_cmd and str(cli_cmd).strip():
            return str(cli_cmd).strip()
    except Exception:
        pass

    # 3. Fallback based on model name
    m_lower = (model or "").lower()
    if "antigravity" in m_lower:
        return 'antigravity chat "{prompt}"'
    return 'agy -p "{prompt}"'


class LocalCLIAuxiliaryClient:
    """OpenAI-compatible client adapter that routes prompts to a local CLI."""

    def __init__(
        self,
        model: Optional[str] = None,
        command_template: Optional[str] = None,
    ):
        self._model = model or "local-cli"
        self.command_template = command_template or get_default_cli_command(model)
        self.api_key = "none"
        self.base_url = "cli://local"
        self._custom_headers = {}

        # Provide .chat.completions.create() structure
        self.completions = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=self.completions)

    def _execute_cli(self, prompt_text: str, timeout: float = 120.0) -> str:
        """Execute the CLI command and return stdout string."""
        cmd_template = self.command_template or get_default_cli_command(self._model)

        if "{prompt}" in cmd_template:
            cmd_str = cmd_template.replace("{prompt}", prompt_text)
            logger.debug("Executing local CLI with prompt template: %s", cmd_str[:120])
            proc = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            # Pass prompt via stdin
            logger.debug("Executing local CLI via stdin: %s", cmd_template)
            proc = subprocess.run(
                cmd_template,
                shell=True,
                input=prompt_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip() or f"Exit code {proc.returncode}"
            logger.error("Local CLI execution failed: %s", err_msg)
            raise RuntimeError(f"Local CLI error ({cmd_template}): {err_msg}")

        return proc.stdout.strip()

    def create(self, **kwargs: Any) -> Any:
        """Synchronous chat.completions.create() implementation."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model") or self._model
        stream = kwargs.get("stream", False)
        timeout = float(kwargs.get("timeout", 120.0))

        prompt_text = _format_messages_to_prompt(messages)
        stdout_output = self._execute_cli(prompt_text, timeout=timeout)

        if stream:
            return self._stream_generator(stdout_output, model)

        message_obj = SimpleNamespace(
            role="assistant",
            content=stdout_output,
            tool_calls=None,
        )
        choice_obj = SimpleNamespace(
            index=0,
            message=message_obj,
            finish_reason="stop",
        )
        usage_obj = SimpleNamespace(
            prompt_tokens=len(prompt_text) // 4,
            completion_tokens=len(stdout_output) // 4,
            total_tokens=(len(prompt_text) + len(stdout_output)) // 4,
        )

        return SimpleNamespace(
            id=f"chatcmpl-local-cli-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[choice_obj],
            usage=usage_obj,
        )

    async def acreate(self, **kwargs: Any) -> Any:
        """Async chat.completions.create() implementation."""
        return await asyncio.to_thread(self.create, **kwargs)

    def _stream_generator(self, text: str, model: str):
        """Yield stream chunks for stream=True callers."""
        chunk_id = f"chatcmpl-local-cli-{uuid.uuid4().hex[:8]}"
        created_time = int(time.time())

        # First chunk with delta content
        delta = SimpleNamespace(role="assistant", content=text)
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        yield SimpleNamespace(
            id=chunk_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[choice],
        )

        # Final chunk with finish_reason
        final_delta = SimpleNamespace()
        final_choice = SimpleNamespace(index=0, delta=final_delta, finish_reason="stop")
        yield SimpleNamespace(
            id=chunk_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[final_choice],
        )

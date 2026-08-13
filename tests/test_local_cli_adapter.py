"""Unit tests for local CLI model provider adapter."""

import os
import unittest
from unittest.mock import MagicMock, patch

from agent.local_cli_adapter import (
    LocalCLIAuxiliaryClient,
    _format_messages_to_prompt,
    get_default_cli_command,
)
from agent.auxiliary_client import resolve_provider_client
from providers import get_provider_profile


class TestLocalCLIAdapter(unittest.TestCase):
    """Test suite for LocalCLIAuxiliaryClient and local-cli provider plugin."""

    def test_provider_profile_registration(self):
        """Verify local-cli provider profile is discoverable by name and aliases."""
        profile = get_provider_profile("local-cli")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "local-cli")
        self.assertEqual(profile.auth_type, "none")

        # Test aliases
        self.assertIs(get_provider_profile("cli"), profile)
        self.assertIs(get_provider_profile("agy"), profile)
        self.assertIs(get_provider_profile("antigravity"), profile)

    def test_format_messages_single_user_message(self):
        """Single user message should return clean content string."""
        messages = [{"role": "user", "content": "Hello agy!"}]
        prompt = _format_messages_to_prompt(messages)
        self.assertEqual(prompt, "Hello agy!")

    def test_format_messages_multiturn(self):
        """Multi-turn messages should format roles clearly."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2=4"},
        ]
        prompt = _format_messages_to_prompt(messages)
        self.assertIn("[System Instructions]\nYou are helpful.", prompt)
        self.assertIn("[User]\nWhat is 2+2?", prompt)
        self.assertIn("[Assistant]\n2+2=4", prompt)

    def test_command_resolution_env(self):
        """HERMES_LOCAL_CLI_COMMAND should override default command."""
        with patch.dict(os.environ, {"HERMES_LOCAL_CLI_COMMAND": "my-custom-cli -p '{prompt}'"}):
            cmd = get_default_cli_command("agy")
            self.assertEqual(cmd, "my-custom-cli -p '{prompt}'")

    def test_command_resolution_fallback(self):
        """Fallback command should adapt to model name."""
        with patch.dict(os.environ, {}, clear=True):
            if "HERMES_LOCAL_CLI_COMMAND" in os.environ:
                del os.environ["HERMES_LOCAL_CLI_COMMAND"]
            cmd_agy = get_default_cli_command("agy")
            cmd_anti = get_default_cli_command("antigravity")
            self.assertIn("agy", cmd_agy)
            self.assertIn("antigravity", cmd_anti)

    @patch("subprocess.run")
    def test_create_template_substitution(self, mock_run):
        """Test completion execution with template substitution."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Response from local agy CLI"
        mock_run.return_value = mock_proc

        client = LocalCLIAuxiliaryClient(model="agy", command_template='echo "{prompt}"')
        resp = client.create(messages=[{"role": "user", "content": "Test prompt"}])

        self.assertEqual(resp.model, "agy")
        self.assertEqual(len(resp.choices), 1)
        self.assertEqual(resp.choices[0].message.content, "Response from local agy CLI")
        self.assertEqual(resp.choices[0].finish_reason, "stop")

        mock_run.assert_called_once()
        call_args, call_kwargs = mock_run.call_args
        self.assertIn("Test prompt", call_args[0])

    @patch("subprocess.run")
    def test_create_stdin_pipe(self, mock_run):
        """Test completion execution passing prompt via stdin when no {prompt} placeholder."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Response via stdin"
        mock_run.return_value = mock_proc

        client = LocalCLIAuxiliaryClient(model="antigravity", command_template="antigravity chat")
        resp = client.create(messages=[{"role": "user", "content": "Hello antigravity"}])

        self.assertEqual(resp.choices[0].message.content, "Response via stdin")
        mock_run.assert_called_once()
        _, call_kwargs = mock_run.call_args
        self.assertEqual(call_kwargs.get("input"), "Hello antigravity")

    @patch("subprocess.run")
    def test_create_stream(self, mock_run):
        """Test streaming mode generator."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Streamed answer"
        mock_run.return_value = mock_proc

        client = LocalCLIAuxiliaryClient(model="agy")
        chunks = list(client.create(messages=[{"role": "user", "content": "Hi"}], stream=True))

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].choices[0].delta.content, "Streamed answer")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

    @patch("subprocess.run")
    def test_create_failure(self, mock_run):
        """Test RuntimeError handling on non-zero returncode."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "CLI command not found"
        mock_run.return_value = mock_proc

        client = LocalCLIAuxiliaryClient(model="agy", command_template="invalid_cmd")
        with self.assertRaises(RuntimeError) as ctx:
            client.create(messages=[{"role": "user", "content": "Hi"}])
        self.assertIn("CLI command not found", str(ctx.exception))

    def test_resolve_provider_client_local_cli(self):
        """Verify resolve_provider_client routes local-cli, agy, and antigravity correctly."""
        for prov_name in ("local-cli", "cli", "agy", "antigravity"):
            client, resolved_model = resolve_provider_client(prov_name)
            self.assertIsNotNone(client)
            self.assertTrue(hasattr(client, "chat"))
            self.assertTrue(hasattr(client.chat.completions, "create"))


if __name__ == "__main__":
    unittest.main()

"""Antigravity / agy Local Bridge provider profile.

Connects to local Antigravity API Bridge Server (e.g. http://127.0.0.1:8000/v1)
which translates OpenAI ChatCompletions REST API calls to local CLI execution.
"""

from providers import register_provider
from providers.base import ProviderProfile


class AntigravityProfile(ProviderProfile):
    """Antigravity / agy Local Bridge provider profile."""

    pass


antigravity = AntigravityProfile(
    name="antigravity",
    aliases=("agy", "agy-cli", "local-bridge", "antigravity-bridge"),
    display_name="Antigravity Local Bridge",
    description="Local OpenAI REST API Bridge for antigravity / agy (http://127.0.0.1:8000/v1)",
    auth_type="api_key",
    env_vars=("ANTIGRAVITY_API_KEY", "OPENAI_API_KEY"),
    base_url="http://127.0.0.1:8000/v1",
    fallback_models=("antigravity", "agy"),
    default_max_tokens=65536,
)

register_provider(antigravity)

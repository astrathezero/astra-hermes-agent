"""Local CLI model provider profile.

Executes models via a local CLI tool (e.g. agy -p "{prompt}" or antigravity chat "{prompt}").
Does not require an API key or remote endpoint.
"""

from providers import register_provider
from providers.base import ProviderProfile


class LocalCLIProfile(ProviderProfile):
    """Local CLI model provider profile."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Models are managed locally by the CLI tool."""
        return ["local-cli", "agy", "antigravity"]


local_cli = LocalCLIProfile(
    name="local-cli",
    aliases=("cli", "agy", "antigravity", "local-command"),
    display_name="Local CLI",
    description="Local CLI model provider (e.g. agy -p \"{prompt}\" or antigravity chat \"{prompt}\")",
    auth_type="none",
    base_url="cli://local",
    supports_health_check=False,
    fallback_models=("local-cli", "agy", "antigravity"),
)

register_provider(local_cli)

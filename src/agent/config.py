"""Configuración de la aplicación, leída desde variables de entorno / `.env`."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: SecretStr
    anthropic_model: str = "claude-sonnet-5-5"
    checkpoint_db: str = "checkpoints.db"
    recursion_limit: int = 10
    max_context_messages: int = 20
    # None = no se envía. Los modelos actuales rechazan valores no default.
    temperature: float | None = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # los valores vienen del entorno

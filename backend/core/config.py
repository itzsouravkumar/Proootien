from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "Protein Surface Analyzer"
    debug: bool = True

    redis_url: Optional[str] = None

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

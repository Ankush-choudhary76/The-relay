"""Configuration settings for The Relay pipeline and services."""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq API Configuration
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Relay Server Configuration
    relay_host: str = "127.0.0.1"
    relay_port: int = 8000
    environment: str = "development"
    log_level: str = "INFO"

    # Provider Selection
    provider_type: Literal["groq", "standin"] = "standin"
    standin_base_url: str = "http://127.0.0.1:8001"

    # Resiliency & Timeouts
    provider_timeout: float = 15.0  # seconds
    max_retries: int = 3
    retry_base_delay: float = 1.0  # seconds
    retry_max_delay: float = 10.0  # seconds
    max_concurrent_requests: int = 5
    queue_max_size: int = 50
    request_deadline: float = 20.0  # total end-to-end deadline in seconds

    # Rate Limiting (Groq Free Tier baselines)
    client_rpm_limit: int = 30
    client_tpm_limit: int = 8000
    client_rpd_limit: int = 1000
    client_tpd_limit: int = 200000

    # Retrieval Defaults
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 90
    top_k_passages: int = 5
    enable_reranking: bool = True
    evidence_threshold: float = 0.35  # minimum BM25 normalized score
    high_confidence_threshold: float = 0.60
    budget_exhaustion_safety_margin: float = 0.85  # trigger reduced retrieval when at 85% of limit

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    papers_dir: Path = data_dir / "papers"
    processed_dir: Path = data_dir / "processed"
    measurements_dir: Path = data_dir / "measurements"
    reports_dir: Path = project_root / "reports"
    results_csv_path: Path = project_root / "results.csv"


settings = Settings()


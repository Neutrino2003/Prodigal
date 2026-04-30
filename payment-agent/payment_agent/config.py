import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    payment_api_base_url: str = os.getenv("PAYMENT_API_BASE_URL", "https://payments.internal.prodigal.ai")
    max_verification_attempts: int = _as_int("MAX_VERIFICATION_ATTEMPTS", 3)
    llm_model: str = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
    request_timeout_seconds: int = _as_int("REQUEST_TIMEOUT_SECONDS", 10)
    payment_timeout_seconds: int = _as_int("PAYMENT_TIMEOUT_SECONDS", 15)
    lookup_network_retries: int = _as_int("LOOKUP_NETWORK_RETRIES", 1)


settings = Settings()

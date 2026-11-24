import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    ch_url: str = os.getenv("CH_URL", "http://localhost:8123")
    ch_user: str = os.getenv("CH_USER", "default")
    ch_password: str = os.getenv("CH_PASSWORD", "")
    ch_database: str = os.getenv("CH_DATABASE", "wms_dist")
    table: str = os.getenv("IPDR_TABLE", "ntc_ssl_log")

    max_exec_seconds: int = _int("IPDR_MAX_EXEC_SECONDS", 600)
    max_result_rows: int = _int("IPDR_MAX_RESULT_ROWS", 5000)
    max_rows_to_read: int = _int("IPDR_MAX_ROWS_TO_READ", 0)
    max_memory_bytes: int = _int("IPDR_MAX_MEMORY_BYTES", 8_000_000_000)

    default_provider: str = os.getenv("IPDR_PROVIDER", "anthropic")
    default_model: str = os.getenv("IPDR_MODEL", "claude-opus-5-5")
    default_base_url: str = os.getenv("IPDR_BASE_URL", "")
    server_api_key: str = os.getenv("IPDR_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
    max_agent_turns: int = _int("IPDR_MAX_AGENT_TURNS", 12)
    review_pass: str = os.getenv("IPDR_REVIEW_PASS", "auto")

    store_path: str = os.getenv("IPDR_STORE_PATH", str(ROOT / "data" / "ipdr.sqlite3"))

    # 0.0.0.0 accepts connections on every interface. 127.0.0.1 is this machine only.
    host: str = os.getenv("IPDR_HOST", "0.0.0.0")
    port: int = _int("IPDR_PORT", 8000)

    timezone: str = "Asia/Karachi"

    @property
    def full_table(self) -> str:
        return f"{self.ch_database}.{self.table}"


settings = Settings()

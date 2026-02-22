"""Common project-level paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY_DB_PATH = PROJECT_ROOT / ".agent_memory.db"


def get_default_memory_db_path() -> Path:
    """Return the unified SQLite memory DB path."""
    return DEFAULT_MEMORY_DB_PATH


def resolve_kline_db_path(workspace_dir: Path) -> Path:
    """Resolve K-line DB path.

    K-line data is unified into the same SQLite DB as session/memory data.
    """
    _ = workspace_dir
    return DEFAULT_MEMORY_DB_PATH

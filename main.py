"""OpenJarvis — Root Entry Point.

Allows launching OpenJarvis directly via:
    python main.py [COMMAND] [ARGS]...

Delegates cleanly to the canonical OpenJarvis CLI runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_windows_console_utf8() -> None:
    """Ensure Windows console streams use UTF-8 encoding without crashing."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (AttributeError, OSError):
                    pass


def _check_virtual_env() -> None:
    """Warn if running outside a virtual environment."""
    is_venv = (
        sys.prefix != sys.base_prefix
        or hasattr(sys, "real_prefix")
        or bool(os.environ.get("VIRTUAL_ENV"))
    )
    if not is_venv:
        if sys.platform == "win32":
            sys.stderr.write(
                "To activate the repository virtual environment, run:\n"
                "  .\\.venv\\Scripts\\Activate.ps1   (PowerShell)\n"
                "  .\\.venv\\Scripts\\activate.bat   (Command Prompt)\n\n"
            )
        else:
            sys.stderr.write(
                "To activate the repository virtual environment, run:\n"
                "  source .venv/bin/activate\n\n"
            )


def _load_env_file(repo_root: Path) -> None:
    """Load key-value pairs from .env into os.environ if not already set."""
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (
                        val.startswith("'") and val.endswith("'")
                    ):
                        val = val[1:-1]
                    if key and val and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass


def main() -> None:
    """Execute OpenJarvis through the canonical CLI interface."""
    _ensure_windows_console_utf8()
    _check_virtual_env()

    repo_root = Path(__file__).resolve().parent
    _load_env_file(repo_root)

    # Defaults for local Ollama + The Deep View exclusive operation
    os.environ.setdefault("OPENJARVIS_DEFAULT_ENGINE", "ollama")
    os.environ.setdefault("OPENJARVIS_EXCLUSIVE_SOURCE", "thedeepview")
    os.environ.setdefault("OPENJARVIS_WEB_SEARCH_ENGINE", "deep_view")
    os.environ.setdefault("JARVIS_SKIP_MODEL_PICK", "1")
    os.environ.setdefault("JARVIS_SKIP_RUNTIME_PANEL", "1")
    os.environ.setdefault("JARVIS_NUM_CTX", "16384")
    os.environ.setdefault("JARVIS_NUM_GPU", "1")
    os.environ.setdefault("OLLAMA_MODEL", "gemma4:latest")
    os.environ.setdefault("JARVIS_MODEL", "gemma4:latest")

    # When invoked as bare `python main.py`, launch the interactive chat session
    if len(sys.argv) == 1:
        sys.argv.append("chat")

    # Ensure src directory is importable if running from source checkout
    src_dir = repo_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from openjarvis.cli import main as cli_main
    except ImportError as exc:
        sys.stderr.write(
            f"Error: Unable to import OpenJarvis runtime components:\n  {exc}\n\n"
            "Ensure dependencies are installed with:\n"
            "  uv sync\n"
        )
        sys.exit(1)

    try:
        cli_main()
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted by user.\n")
        sys.exit(130)


if __name__ == "__main__":
    main()

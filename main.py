"""OpenJarvis — Root Entry Point.

Allows launching OpenJarvis directly via:
    python main.py [COMMAND] [ARGS]...

Supports Groq as the primary online inference engine with automatic
local fallback to Ollama (gemma4:latest, 16K context window).
Exclusive web research is locked to The Deep View (https://www.thedeepview.com).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
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
    """Load key-value pairs from .env into os.environ (overwrites existing values)."""
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
                    if key:
                        os.environ[key] = val
    except Exception:
        pass


def _probe_groq_online(api_key: str | None) -> tuple[bool, str | None]:
    """Test if Groq API is online, reachable, and authenticated."""
    if not api_key or not api_key.strip():
        return False, None
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "User-Agent": "OpenJarvis/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                model_ids = [
                    m["id"]
                    for m in payload.get("data", [])
                    if isinstance(m, dict) and "id" in m
                ]
                preferred = [
                    "openai/gpt-oss-120b",
                    "qwen/qwen3.8-27b",
                    "openai/gpt-oss-20b",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                ]
                selected = os.environ.get("GROQ_MODEL")
                if not selected:
                    for pref in preferred:
                        if pref in model_ids:
                            selected = pref
                            break
                if not selected and model_ids:
                    chat_models = [
                        m
                        for m in model_ids
                        if not m.startswith("whisper") and "guard" not in m.lower()
                    ]
                    selected = chat_models[0] if chat_models else model_ids[0]
                return True, selected or "openai/gpt-oss-120b"
    except Exception:
        pass
    return False, None


def main() -> None:
    """Execute OpenJarvis through the canonical CLI interface."""
    _ensure_windows_console_utf8()
    _check_virtual_env()

    repo_root = Path(__file__).resolve().parent
    _load_env_file(repo_root)

    # Detect online Groq connectivity vs. local Ollama fallback
    groq_api_key = os.environ.get("GROQ_API_KEY")
    is_groq_online, groq_model = _probe_groq_online(groq_api_key)

    if is_groq_online and groq_model:
        os.environ["OPENJARVIS_DEFAULT_ENGINE"] = "groq"
        os.environ["JARVIS_MODEL"] = groq_model
        os.environ["GROQ_MODEL"] = groq_model
        sys.stdout.write(
            f"\n⚡ [Online Mode] Connected to Groq API. Engine: Groq | Model: {groq_model}\n"
        )
    else:
        os.environ["OPENJARVIS_DEFAULT_ENGINE"] = "ollama"
        os.environ["OLLAMA_MODEL"] = "gemma4:latest"
        os.environ["JARVIS_MODEL"] = "gemma4:latest"
        os.environ["JARVIS_NUM_CTX"] = "16384"
        os.environ["JARVIS_NUM_GPU"] = "1"
        if groq_api_key:
            sys.stdout.write(
                "\n⚡ [Offline / Local Fallback] Groq unreachable or offline. Falling back to local Ollama (gemma4:latest, 16K context).\n"
            )
        else:
            sys.stdout.write(
                "\n⚡ [Local Mode] Using local Ollama (gemma4:latest, 16K context).\n"
            )

    # Defaults for The Deep View exclusive web research
    os.environ.setdefault("OPENJARVIS_EXCLUSIVE_SOURCE", "thedeepview")
    os.environ.setdefault("OPENJARVIS_WEB_SEARCH_ENGINE", "deep_view")
    os.environ.setdefault("JARVIS_SKIP_MODEL_PICK", "1")
    os.environ.setdefault("JARVIS_SKIP_RUNTIME_PANEL", "1")

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

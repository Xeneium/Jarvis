"""Test suite for root main.py entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_main_py_exists():
    """Verify main.py exists at repository root."""
    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "main.py"
    assert main_py.is_file(), "main.py must exist at the repository root"


def test_main_py_help_invocation():
    """Verify python main.py --help succeeds with code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "main.py"
    result = subprocess.run(
        [sys.executable, str(main_py), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(repo_root),
        timeout=15,
    )
    assert result.returncode == 0, f"Expected 0, got {result.returncode}. Stderr: {result.stderr}"
    assert "OpenJarvis" in result.stdout
    assert "doctor" in result.stdout
    assert "ask" in result.stdout
    assert "chat" in result.stdout


def test_main_py_version_invocation():
    """Verify python main.py --version succeeds with code 0."""
    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "main.py"
    result = subprocess.run(
        [sys.executable, str(main_py), "--version"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=15,
    )
    assert result.returncode == 0
    assert "jarvis, version" in result.stdout


def test_load_env_file(tmp_path: Path, monkeypatch):
    """Verify _load_env_file correctly parses key=value from .env."""
    import os
    from main import _load_env_file

    env_path = tmp_path / ".env"
    env_path.write_text("TEST_OPENJARVIS_KEY=my_secret_token_123\n# comment\nOTHER_KEY=\"quoted_val\"\n")

    monkeypatch.delenv("TEST_OPENJARVIS_KEY", raising=False)
    monkeypatch.delenv("OTHER_KEY", raising=False)

    _load_env_file(tmp_path)

    assert os.environ.get("TEST_OPENJARVIS_KEY") == "my_secret_token_123"
    assert os.environ.get("OTHER_KEY") == "quoted_val"


def test_main_runtime_panel_defaults(monkeypatch):
    """Verify main() sets default runtime environment variables when absent."""
    import os
    import sys
    from unittest.mock import patch
    import main as main_module

    for k in (
        "OPENJARVIS_DEFAULT_ENGINE",
        "OPENJARVIS_EXCLUSIVE_SOURCE",
        "OPENJARVIS_WEB_SEARCH_ENGINE",
        "JARVIS_SKIP_MODEL_PICK",
        "JARVIS_SKIP_RUNTIME_PANEL",
        "JARVIS_NUM_CTX",
        "JARVIS_NUM_GPU",
        "OLLAMA_MODEL",
        "JARVIS_MODEL",
    ):
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])
    with patch("openjarvis.cli.main"):
        main_module.main()

    assert os.environ.get("OPENJARVIS_DEFAULT_ENGINE") == "ollama"
    assert os.environ.get("OPENJARVIS_EXCLUSIVE_SOURCE") == "thedeepview"
    assert os.environ.get("JARVIS_SKIP_RUNTIME_PANEL") == "1"
    assert os.environ.get("JARVIS_NUM_CTX") == "16384"
    assert os.environ.get("JARVIS_NUM_GPU") == "1"
    assert os.environ.get("OLLAMA_MODEL") == "gemma4:latest"
    assert os.environ.get("JARVIS_MODEL") == "gemma4:latest"


def test_resolve_chat_cli_model_rejects_qwen_for_ollama(monkeypatch):
    """Verify that resolve_chat_cli_model rejects qwen and prefers gemma4:latest for Ollama."""
    from unittest.mock import MagicMock, patch
    from openjarvis.cli._model_switch import resolve_chat_cli_model
    from openjarvis.core.config import JarvisConfig

    cfg = JarvisConfig()
    cfg.intelligence.default_model = ""
    eng = MagicMock()
    with (
        patch("openjarvis.engine.discover_engines", return_value=[("ollama", eng)]),
        patch("openjarvis.engine.discover_models", return_value={"ollama": ["qwen3:8b", "qwen2.5:7b"]}),
    ):
        m = resolve_chat_cli_model(
            console=MagicMock(),
            config=cfg,
            engine=eng,
            engine_name="ollama",
            cli_model=None,
            chat_variant="chat",
        )
    assert m == "gemma4:latest"
    assert "qwen" not in m.lower()


def test_resolve_chat_cli_model_selects_gemma4_if_available(monkeypatch):
    """Verify that resolve_chat_cli_model picks gemma4 if present in discovered models."""
    from unittest.mock import MagicMock, patch
    from openjarvis.cli._model_switch import resolve_chat_cli_model
    from openjarvis.core.config import JarvisConfig

    cfg = JarvisConfig()
    cfg.intelligence.default_model = ""
    eng = MagicMock()
    with (
        patch("openjarvis.engine.discover_engines", return_value=[("ollama", eng)]),
        patch("openjarvis.engine.discover_models", return_value={"ollama": ["qwen3:8b", "gemma4:latest"]}),
    ):
        m = resolve_chat_cli_model(
            console=MagicMock(),
            config=cfg,
            engine=eng,
            engine_name="ollama",
            cli_model=None,
            chat_variant="chat",
        )
    assert m == "gemma4:latest"


# Open WebUI starts from the current config/.env, never from what an older tmux server still holds: the session runs
# scripts/openwebui_session.sh, which clears the inherited environment and reads config/.env itself. Secret values never
# appear in the tmux command (a process listing shows it), and Open WebUI's own default connection gets no model key.
import os
import re
import subprocess
from pathlib import Path

from memento.config.paths import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "openwebui_session.sh"


def _fake_install(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "memento"
    (root / "config").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "config" / ".env").write_text("WEBUI_SECRET_KEY=fresh-secret\nDATABASE_URL=postgresql://u:fresh@db/x\n"
                                          "OPENAI_API_KEY=model-key\nDOMAIN_NAME=generic\n")
    seen = tmp_path / "seen.env"
    stub = root / ".venv" / "bin" / "open-webui"
    stub.write_text(f"#!/usr/bin/env bash\nenv > '{seen}'\necho \"args: $*\" >> '{seen}'\n")
    stub.chmod(0o755)
    return root, seen


def test_the_session_reads_the_current_env_file_and_drops_what_the_tmux_server_held(tmp_path):
    root, seen = _fake_install(tmp_path)
    stale = {**os.environ, "WEBUI_SECRET_KEY": "stale-secret", "DATABASE_URL": "postgresql://u:stale@db/x",
             "LEFT_OVER_FROM_AN_OLD_START": "1", "OPENAI_API_KEY": "stale-key", "HOME": str(tmp_path)}
    r = subprocess.run(["bash", str(SCRIPT), str(root), "127.0.0.1", "8099", str(tmp_path / "ow.log")],
                       env=stale, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-1500:]
    env = dict(l.split("=", 1) for l in seen.read_text().splitlines() if "=" in l and not l.startswith("args:"))
    assert env["WEBUI_SECRET_KEY"] == "fresh-secret" and env["DATABASE_URL"].endswith(":fresh@db/x")
    assert "LEFT_OVER_FROM_AN_OLD_START" not in env
    assert "OPENAI_API_KEY" not in env                 # Open WebUI's own default connection gets no model key
    assert env["DOMAIN_NAME"] == "generic" and env["DATA_DIR"] == f"{tmp_path}/.open-webui"
    assert env["WEBUI_NAME"] == "Memento"
    assert "args: serve --host 127.0.0.1 --port 8099" in seen.read_text()


def test_services_starts_open_webui_through_the_session_script_without_secrets_in_the_command():
    src = (PROJECT_ROOT / "scripts" / "services.sh").read_text()
    body = src[src.index("start_openwebui() {"):src.index("\n}\n", src.index("start_openwebui() {"))]
    cmd = body[body.index("tmux new-session"):body.index("local retries")]
    assert "openwebui_session.sh" in cmd
    assert not re.search(r"DATABASE_URL|WEBUI_SECRET_KEY|API_KEY|PASSWORD", cmd), cmd

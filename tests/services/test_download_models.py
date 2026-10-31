# scripts/download_models.sh finds the `hf` command of the project's own environment (INSTALL.md step 2) when it is not on
# PATH: a fresh account has no `hf` of its own, and on Ubuntu 24.04 a plain `pip install` outside a venv is refused.
import shutil
import subprocess

from memento.config.paths import PROJECT_ROOT


def _project(tmp_path, with_venv_hf: bool):
    proj = tmp_path / "proj"
    (proj / "scripts").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "scripts/download_models.sh", proj / "scripts/download_models.sh")
    calls = tmp_path / "calls.txt"
    if with_venv_hf:
        hf = proj / ".venv/bin/hf"
        hf.parent.mkdir(parents=True)
        hf.write_text(f'#!/bin/sh\necho "$@" >> {calls}\nmkdir -p "$4" && echo weights > "$4/model.safetensors"\n')
        hf.chmod(0o755)
    return proj, calls


def _run(proj, tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "MODELS_DIR": str(tmp_path / "models")}
    return subprocess.run(["bash", "scripts/download_models.sh", "embedder"], cwd=proj, env=env, capture_output=True,
                          text=True, timeout=60)


def test_uses_the_project_environments_hf_when_none_is_on_path(tmp_path):
    proj, calls = _project(tmp_path, with_venv_hf=True)
    r = _run(proj, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls.read_text().split() == ["download", "Qwen/Qwen3-VL-Embedding-8B", "--local-dir",
                                         str(tmp_path / "models/Qwen3-VL-Embedding-8B")]


def test_without_any_hf_it_points_to_the_readme_step(tmp_path):
    proj, _ = _project(tmp_path, with_venv_hf=False)
    r = _run(proj, tmp_path)
    assert r.returncode != 0
    assert "requirements.txt" in r.stdout and "pip install 'huggingface_hub[cli]'" not in r.stdout

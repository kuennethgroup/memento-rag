# The install paths agree with each other: README step 1 names every package a fresh Ubuntu needs, step 2 clones the
# published repository, the one-click installer follows README steps 2 to 9 (Python 3.12, the answer model of
# config/.env, the proxy on 8443), and the integration guide's Valve table shows today's defaults.
import subprocess

from memento.config.paths import PROJECT_ROOT

README = (PROJECT_ROOT / "README.md").read_text()
ONE = (PROJECT_ROOT / "one_click_setup.sh").read_text()


def test_step_1_names_git_and_the_python_venv_package():
    step1 = README[README.index("### Step 1"):README.index("### Step 2")]
    apt = next(l for l in step1.splitlines() if l.startswith("sudo apt install podman"))
    assert "git" in apt.split() and "python3.12-venv" in apt.split()


def test_step_2_clones_the_published_repository():
    step2 = README[README.index("### Step 2"):README.index("### Step 3")]
    assert "git clone https://github.com/kuennethgroup/memento-rag" in step2 and "<repo-url>" not in step2


def test_the_tunnel_step_says_the_local_port_must_match_files_url():
    step7 = README[README.index("### Step 7"):README.index("### Step 8")]
    assert "ssh -L 8443:localhost:8443" in step7 and "local port" in step7
    assert "`FILES_URL`" in step7 and "Valve" in step7


def test_the_one_click_installer_follows_the_readme():
    assert subprocess.run(["bash", "-n", str(PROJECT_ROOT / "one_click_setup.sh")]).returncode == 0
    assert "python3.12 -m venv" in ONE
    for var in ("REASONING_URL", "REASONING_MODEL_NAME", "REASONING_API_KEY", "DOMAIN_NAME", "WEBUI_SECRET_KEY"):
        assert var in ONE, var
    assert "localhost:8443" in ONE and "localhost:8086" not in ONE
    assert "files openwebui proxy" in ONE and "OPENAI_API_KEY" not in ONE


def test_the_integration_guide_shows_todays_valve_defaults():
    ow = (PROJECT_ROOT / "openwebui/README.md").read_text()
    row = next(l for l in ow.splitlines() if l.startswith("| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY_ENV`"))
    assert "REASONING_URL" in row and "REASONING_API_KEY" in row and "Academic Cloud" not in row
    files = next(l for l in ow.splitlines() if l.startswith("| `FILES_URL`"))
    assert "8443/files" in files

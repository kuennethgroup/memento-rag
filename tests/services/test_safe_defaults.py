# A new installation is reachable only from the server itself, and a stranger who reaches it cannot use it at once:
# the proxy binds to 127.0.0.1 unless PROXY_BIND opens it, and trial's sign-up lets new accounts wait for an admin's
# approval unless --open-signup is given (as for a moderated study).
import importlib.util
import re
import subprocess
import sys

import yaml

from memento.config.paths import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("so_safe", PROJECT_ROOT / "scripts/setup_openwebui.py")
so = importlib.util.module_from_spec(spec)
spec.loader.exec_module(so)


def test_new_accounts_wait_for_approval_unless_sign_up_is_opened():
    assert so.signup_settings("g1", open_signup=False) == {"ENABLE_SIGNUP": True, "DEFAULT_USER_ROLE": "pending",
                                                           "DEFAULT_GROUP_ID": "g1"}
    assert so.signup_settings("g1", open_signup=True)["DEFAULT_USER_ROLE"] == "user"


def test_the_trial_command_offers_open_sign_up():
    out = subprocess.run([sys.executable, "scripts/setup_openwebui.py", "--help"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "--open-signup" in out.stdout


def test_the_proxy_binds_to_the_server_itself_by_default():
    caddy = (PROJECT_ROOT / "docker/Caddyfile").read_text()
    site = caddy[caddy.index(":{$PROXY_PORT:8443} {"):]
    assert re.search(r"(?m)^\s*bind \{\$PROXY_BIND:127\.0\.0\.1\}\s*$", site)
    proxy = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())["services"]["proxy"]
    assert proxy["environment"]["PROXY_BIND"] == "${PROXY_BIND:-127.0.0.1}"


def test_the_readme_says_how_to_open_access_and_sign_up():
    readme = (PROJECT_ROOT / "README.md").read_text()
    access = readme[readme.index("## Access from other computers"):]
    assert "PROXY_BIND=0.0.0.0" in access.split("\n---")[0]
    assert 'PROXY_BIND="0.0.0.0 ::"' in access.split("\n---")[0]            # IPv6 as well (quoted: .env is sourced)
    step8 = readme[readme.index("### Step 8"):readme.index("### Step 9")]
    assert "--open-signup" in step8 and "approv" in step8


def test_the_file_gate_admits_only_approved_accounts():
    # the proxy asks Open WebUI before every /files request; the endpoint it asks must refuse pending accounts
    import importlib.util
    import pytest
    caddy = (PROJECT_ROOT / "docker/Caddyfile").read_text()
    uri = re.search(r"forward_auth \S+ \{\s*uri (\S+)", caddy).group(1)
    assert uri == "/api/v1/users/user/info"
    spec = importlib.util.find_spec("open_webui")
    if spec is None:
        pytest.skip("Open WebUI not installed")
    root = __import__("pathlib").Path(spec.submodule_search_locations[0])
    users = (root / "routers/users.py").read_text()
    route = users[users.index("@router.get('/user/info'"):]
    assert "Depends(get_verified_user)" in route[:route.index("\n\n")]
    roles = re.search(r"VERIFIED_USER_ROLES = (\{[^}]*\})", (root / "utils/auth.py").read_text()).group(1)
    assert "pending" not in roles and "'user'" in roles

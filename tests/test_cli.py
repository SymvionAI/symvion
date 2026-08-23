from pathlib import Path

from click.testing import CliRunner

from symvion.cli import create_symvion, main, scaffold_project


def test_lazy_public_export():
    from symvion import TenantConfig

    assert TenantConfig.__name__ == "TenantConfig"


def test_create_symvion_scaffolds_without_install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(create_symvion, ["my-app", "--no-install"])

    assert result.exit_code == 0, result.output
    root = tmp_path / "my-app"
    assert (root / "main.py").exists()
    assert (root / ".env").exists()
    assert (root / "config" / "policies.yaml").exists()
    assert (root / "agents" / "custom_agent.py").exists()
    assert (root / "tools" / "custom_tool.py").exists()
    assert (root / "agents" / "__init__.py").exists()
    assert (root / "tools" / "__init__.py").exists()
    assert not (root / "data" / "__init__.py").exists()
    assert not (root / "logs" / "__init__.py").exists()
    assert not (root / ".venv").exists()

    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "python-dotenv" in requirements
    assert requirements.splitlines()[0].startswith("symvion")


def test_init_is_alias_for_create(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["init", "demo", "--no-install"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "demo" / "main.py").exists()


def test_create_subcommand(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["create", "from-group", "--no-install"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "from-group" / "requirements.txt").exists()


def test_refuses_existing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "exists").mkdir()
    runner = CliRunner()
    result = runner.invoke(create_symvion, ["exists", "--no-install"])

    assert result.exit_code != 0
    combined = f"{result.output}{result.stderr or ''}"
    assert "already exists" in combined


def test_scaffold_pins_requirements(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert scaffold_project("pinned", install=False) == 0
    text = Path("pinned/requirements.txt").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert first == "symvion" or first.startswith("symvion==")
    assert ".venv/" in Path("pinned/.gitignore").read_text(encoding="utf-8")


def test_install_passes_absolute_requirements_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert scaffold_project("proj", install=False) == 0

    recorded = {}

    def fake_check_call(cmd, cwd=None):
        recorded["cmd"] = cmd
        recorded["cwd"] = cwd
        return 0

    monkeypatch.setattr("symvion.cli.subprocess.check_call", fake_check_call)
    monkeypatch.setattr("symvion.cli.venv.create", lambda *a, **k: None)

    from symvion.cli import _install_project, _venv_python

    python = _venv_python(tmp_path / "proj" / ".venv")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")

    assert _install_project(Path("proj")) is True
    req = Path(recorded["cmd"][-1])
    assert req.is_absolute()
    assert req.name == "requirements.txt"
    assert req.exists()
    assert Path(recorded["cwd"]).resolve() == (tmp_path / "proj").resolve()

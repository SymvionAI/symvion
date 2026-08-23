import os
import subprocess
import venv
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click

# Scaffolding Templates

MAIN_PY_TEMPLATE = """from symvion import Symvion, TenantConfig
import asyncio
import os

from dotenv import load_dotenv

async def main():
    load_dotenv()
    
    # Option 1: Hardcoded Configuration (Fast Prototyping)
    config = TenantConfig(
        env_vars={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")},
        tenant_id="acme-corp",
        llm_provider="openai",
        llm_config={"model": "gpt-4", "temperature": 0.7},
        enabled_agents=["researcher", "reviewer"]
    )
    
    # Option 2: Enterprise Policy Loading (Commented Out)
    # config = ConfigRegistry.load_tenant_config(
    #     path="config/policies.yaml", 
    #     tenant_id="acme-corp"
    # )
    
    runtime = Symvion(config)
    
    print("Symvion AI Runtime Initialized.")
    print("-" * 30)
    
    # Start a chat session
    response = await runtime.chat(
        tenant="acme-corp",
        session_id="session-123",
        message="Hello, how can you help me today?"
    )
    
    print(f"Agent Response: {response['data']}")

if __name__ == "__main__":
    asyncio.run(main())
"""

DOTENV_TEMPLATE = """# Symvion Environment Configuration
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
SYM_LOG_LEVEL=INFO
"""

POLICIES_YAML_TEMPLATE = """# Symvion Enterprise Policies
tenants:
  acme-corp:
    governance:
      pii_redaction: true
      fraud_scoring: true
    guardrails:
      max_tokens_per_turn: 4000
      recursion_limit: 20
      enable_thought_auditing: true
    agents:
      researcher:
        model: "gpt-4"
        temperature: 0.2
      reviewer:
        model: "o1-preview"
        max_completion_tokens: 2000
"""

EXAMPLE_AGENT_TEMPLATE = """from symvion.agents.base import BaseAgent
from symvion import tool

class CustomResearchAgent(BaseAgent):
    \"\"\"
    An example enterprise agent for research tasks.
    \"\"\"
    def __init__(self, tenant_id, config):
        super().__init__(tenant_id, config)
        self.name = "researcher"
        self.description = "Specialized in deep research and documentation."

    async def execute(self, state):
        # Implementation logic here
        return state
"""

EXAMPLE_TOOL_TEMPLATE = '''from symvion import tool

@tool
def search_internal_docs(query: str) -> str:
    """
    Searches the internal enterprise knowledge base.
    """
    return f"Search results for: {query} (Simulated)"
'''

GITIGNORE_TEMPLATE = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
.venv/
venv/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Symvion
.env
logs/*.log
data/processed/
.pytest_cache/
"""

DATA_SAMPLE_TEMPLATE = """# Acme Enterprise Knowledge Base
This is a sample data file. In a real project, you might place PDFs, 
manuals, or raw text files here for your Retrieval Agents to index.
"""

LOG_HELP_TEMPLATE = """# Symvion Execution Logs
This directory will store execution traces and thought audits.
By default, the runtime logs to stdout, but you can configure 
file handlers in your logging setup.
"""

def _no_install_option():
    return click.option(
        "--no-install",
        is_flag=True,
        help="Scaffold only; skip creating a virtualenv and installing dependencies.",
    )


def _symvion_requirement() -> str:
    try:
        return f"symvion=={version('symvion')}"
    except PackageNotFoundError:
        return "symvion"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _activate_hint() -> str:
    if os.name == "nt":
        return ".venv\\Scripts\\activate"
    return "source .venv/bin/activate"


def _next_steps(project_name: str, *, installed: bool) -> str:
    activate = _activate_hint()
    lines = [
        "Next steps:",
        f"  cd {project_name}",
    ]
    if not installed:
        lines.append("  python -m venv .venv")
    lines.append(f"  {activate}")
    if not installed:
        lines.append("  python -m pip install -r requirements.txt")
    lines.extend([
        "  # Edit .env with your API key",
        "  python main.py",
    ])
    return "\n".join(lines)


def _install_project(root: Path) -> bool:
    root = root.resolve()
    venv_dir = root / ".venv"
    click.echo("  Creating virtual environment...")
    try:
        venv.create(venv_dir, with_pip=True)
    except Exception as exc:
        click.echo(f"Error: could not create virtualenv: {exc}", err=True)
        return False

    python = _venv_python(venv_dir)
    if not python.exists():
        click.echo(f"Error: virtualenv python not found at {python}", err=True)
        return False

    requirements = root / "requirements.txt"
    click.echo("  Installing dependencies (includes pinned symvion)...")
    try:
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "-r", str(requirements)],
            cwd=str(root),
        )
    except subprocess.CalledProcessError:
        click.echo(
            "Error: pip install failed. The project files were created; "
            "run `python -m pip install -r requirements.txt` inside the venv.",
            err=True,
        )
        return False
    return True


def scaffold_project(project_name: str, *, install: bool = True) -> int:
    """Create a Symvion project. Returns a process exit code."""
    root = Path(project_name)

    if root.exists():
        click.echo(f"Error: Directory '{project_name}' already exists.", err=True)
        return 1

    click.echo(f"Creating Symvion project: {project_name}...")

    package_dirs = [root / "agents", root / "tools"]
    plain_dirs = [root / "config", root / "data", root / "logs"]

    for d in package_dirs + plain_dirs:
        d.mkdir(parents=True)
        click.echo(f"  Created {d}/")

    for d in package_dirs:
        (d / "__init__.py").touch()

    files = {
        root / "main.py": MAIN_PY_TEMPLATE,
        root / ".env": DOTENV_TEMPLATE,
        root / ".gitignore": GITIGNORE_TEMPLATE,
        root / "config" / "policies.yaml": POLICIES_YAML_TEMPLATE,
        root / "agents" / "custom_agent.py": EXAMPLE_AGENT_TEMPLATE,
        root / "tools" / "custom_tool.py": EXAMPLE_TOOL_TEMPLATE,
        root / "data" / "knowledge_base.txt": DATA_SAMPLE_TEMPLATE,
        root / "logs" / "README.md": LOG_HELP_TEMPLATE,
        root / "requirements.txt": f"{_symvion_requirement()}\npython-dotenv\n",
    }

    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"  Created {path}")

    installed = False
    if install:
        if not _install_project(root):
            click.echo("-" * 30)
            click.echo("Project files were created, but dependency install failed.")
            click.echo(_next_steps(project_name, installed=False))
            return 1
        installed = True

    click.echo("-" * 30)
    click.echo("Success! Your Symvion project is ready.")
    click.echo(_next_steps(project_name, installed=installed))
    return 0


def _run_scaffold(project_name: str, no_install: bool) -> None:
    code = scaffold_project(project_name, install=not no_install)
    if code:
        raise SystemExit(code)


@click.group()
def main():
    """Symvion AI Orchestration Framework CLI."""
    pass


@main.command()
@click.argument("project_name")
@_no_install_option()
def create(project_name, no_install):
    """Create a new Symvion project with a virtualenv and pinned dependencies."""
    _run_scaffold(project_name, no_install)


@main.command()
@click.argument("project_name")
@_no_install_option()
def init(project_name, no_install):
    """Alias for create."""
    _run_scaffold(project_name, no_install)


@click.command("create-symvion")
@click.argument("project_name")
@_no_install_option()
def create_symvion(project_name, no_install):
    """Create a new Symvion project (Python equivalent of create-next-app)."""
    _run_scaffold(project_name, no_install)


if __name__ == "__main__":
    main()

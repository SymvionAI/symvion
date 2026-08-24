import os
import subprocess
import venv
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click

# Scaffolding Templates

MAIN_PY_TEMPLATE = """from symvion import AgentRegistration, Symvion, TenantConfig
import asyncio
import os

from dotenv import load_dotenv

from agents.custom_agent import CustomAgent
from tools.custom_tool import multiply_and_add

async def main():
    load_dotenv()

    config = TenantConfig(
        env_vars={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")},
        tenant_id="acme-corp",
        llm_provider="openai",
        llm_config={"model": "gpt-4", "temperature": 0.7},
        router_type="llm",
        iam_policies={
            "user": ["multiply_and_add"],
            "tenant_admin": ["*"],
        },
    )

    runtime = Symvion(config)
    runtime.register_agent(
        AgentRegistration(
            name="math",
            description="Uses multiply_and_add to multiply two numbers and add a third.",
            system_prompt=(
                "You are a math assistant. When the user asks to multiply and add numbers, "
                "you must call the multiply_and_add tool. Do not compute the result yourself."
            ),
            tools=["multiply_and_add"],
        )
    )

    runtime.register_agent({
        "name": "custom",
        "agent_class": CustomAgent,
        "description": "A simple agent that can perform custom tasks.",
        "system_prompt": (
            "You are a custom assistant. When the user asks to perform custom tasks, "
            "you must call the custom tool. Do not compute the result yourself."
        ),
        "tools": [],
    })

    print("Symvion AI Runtime Initialized.")
    print("-" * 30)

    response = await runtime.chat(
        tenant="acme-corp",
        session_id="session-math-4",
        message="Use the multiply_and_add tool to multiply 2 and 3 and add 4",
    )

    print(f"Agent: {response['agent']}")
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

EXAMPLE_AGENT_TEMPLATE = """from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from symvion.agents.base import BaseAgent
from symvion.core.context import TenantContext
from symvion.tools.base import ToolSafetyWrapper
from symvion.tools.hitl import run_tool_with_hitl
from symvion.utils.helpers import ensure_messages, filter_allowed_tools
from tools.custom_tool import multiply_and_add

class CustomAgent(BaseAgent):
    \"\"\"
    A simple agent that can perform custom tasks.
    \"\"\"
    def __init__(self, tenant_id, config, tools=None):
        super().__init__(tenant_id, config)
        self.name = config.get("name", "custom")
        self.description = config.get(
            "description",
            "A simple agent that can perform custom tasks.",
        )
        self.system_prompt = config.get(
            "system_prompt",
            "You are a custom assistant. When the user asks to perform custom tasks, "
            "you must call the custom tool. Do not compute the result yourself.",
        )
        self.tools = tools or []

    async def execute(
        self,
        context: TenantContext,
        input_data: Dict[str, Any],
        tools: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        message = input_data.get("message", "")
        history = input_data.get("history", [])

        allowed_tools = filter_allowed_tools(context, self.tools)
        llm_with_tools = self.llm.bind_tools(allowed_tools) if allowed_tools else self.llm

        messages = [SystemMessage(content=self.system_prompt)]
        messages.extend(ensure_messages(history[-5:]))
        messages.append(HumanMessage(content=message))

        response = None
        usage = {}
        async for chunk in llm_with_tools.astream(messages, config=config):
            if response is None:
                response = chunk
            else:
                response += chunk
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = chunk.usage_metadata

        tools_called = False
        if getattr(response, "tool_calls", None):
            import json

            tools_called = True
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = dict(tool_call.get("args") or {})
                call_id = tool_call.get("id", f"call_{tool_name}")
                matched_tool = next((t for t in allowed_tools if t.name == tool_name), None)

                async def _exec(name=tool_name, **kwargs):
                    iam = context.metadata.get("iam_policies")
                    fn = next((t for t in allowed_tools if t.name == name), None)
                    if fn is None:
                        return f"Error: Tool {name} not found"
                    t_func = getattr(fn, "func", fn)
                    if hasattr(t_func, "func"):
                        t_func = t_func.func
                    return await ToolSafetyWrapper.invoke(
                        t_func, context, name, kwargs or {}, iam_policies=iam
                    )

                result = await run_tool_with_hitl(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    call_id=call_id,
                    execute=_exec,
                    config=config,
                    tool=matched_tool,
                )
                messages.append(
                    ToolMessage(
                        content=json.dumps(result) if not isinstance(result, str) else result,
                        tool_call_id=tool_call["id"],
                    )
                )

            response = None
            async for chunk in self.llm.astream(messages, config=config):
                if response is None:
                    response = chunk
                else:
                    response += chunk
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = chunk.usage_metadata

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata

        return {
            "agent_response": response.content,
            "agent_type": self.name,
            "token_usage": usage,
            "tools_called": tools_called,
        }
"""

EXAMPLE_TOOL_TEMPLATE = '''from symvion import tool

@tool
def search_internal_docs(query: str) -> str:
    """
    Searches the internal enterprise knowledge base.
    """
    return f"Search results for: {query} (Simulated)"


@tool
def multiply_and_add(a: int, b: int, c: int) -> int:
    """
    Multiplies two numbers and adds a third number and add 2 to the result.
    """
    return a * b + c + 2

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

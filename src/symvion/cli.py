import os
import click
import shutil
from pathlib import Path

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


@click.group()
def main():
    """Symvion AI Orchestration Framework CLI."""
    pass

@main.command()
@click.argument('project_name')
def init(project_name):
    """Initialize a new Symvion AI project structure."""
    root = Path(project_name)
    
    if root.exists():
        click.echo(f"Error: Directory '{project_name}' already exists.")
        return

    click.echo(f"Creating Symvion project: {project_name}...")
    
    # Create directories
    dirs = [
        root / "agents",
        root / "tools",
        root / "config",
        root / "data",
        root / "logs"
    ]
    
    for d in dirs:
        d.mkdir(parents=True)
        (d / "__init__.py").touch()
        click.echo(f"  Created {d}/")

    # Create files
    files = {
        root / "main.py": MAIN_PY_TEMPLATE,
        root / ".env": DOTENV_TEMPLATE,
        root / ".gitignore": GITIGNORE_TEMPLATE,
        root / "config" / "policies.yaml": POLICIES_YAML_TEMPLATE,
        root / "agents" / "custom_agent.py": EXAMPLE_AGENT_TEMPLATE,
        root / "tools" / "custom_tool.py": EXAMPLE_TOOL_TEMPLATE,
        root / "data" / "knowledge_base.txt": DATA_SAMPLE_TEMPLATE,
        root / "logs" / "README.md": LOG_HELP_TEMPLATE,
        root / "requirements.txt": "symvion\npython-dotenv\n"
    }

    for path, content in files.items():
        with open(path, 'w') as f:
            f.write(content)
        click.echo(f"  Created {path}")

    click.echo("-" * 30)
    click.echo("Success! Your Symvion project is ready.")
    click.echo(f"Next steps:\n  cd {project_name}\n  pip install -r requirements.txt\n  python main.py")

if __name__ == "__main__":
    main()

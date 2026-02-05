import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rich.console import Console

load_dotenv()
console = Console()


@dataclass
class AgentConfig:
    """
    Configuration for an agent.

    This makes agents self-describing and allows the dispatcher
    to route based on patterns without hard-coding logic.
    """

    agent_id: str
    name: str
    description: str
    patterns: list[str]
    capabilities: list[str]
    example_queries: list[str]


class BaseAgent(ABC):
    """
    Base class for all agents.

    Each agent must:
    1. Define its configuration via get_config()
    2. Implement process() to handle requests
    """

    def __init__(self, model: str | None = None, verbose: bool = False):
        """Initialize the agent with LLM."""
        self.model_name = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.verbose = verbose
        self.llm = self._init_llm()

    def _init_llm(self) -> ChatGroq:
        """Initialize the Groq LLM client."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        return ChatGroq(
            model=self.model_name,
            api_key=api_key,
            temperature=0.3,
        )

    @classmethod
    @abstractmethod
    def get_config(cls) -> AgentConfig:
        """
        Return the agent's configuration.

        This makes the agent self-describing for routing and discovery.
        """
        pass

    @abstractmethod
    def process(self, query: str) -> dict:
        """
        Process a user query.

        Args:
            query: User's request

        Returns:
            Dictionary with results (format depends on agent)
        """
        pass

    @contextmanager
    def thinking(self, message: str = "Thinking"):
        """Context manager that shows a spinner while LLM is processing."""
        if self.verbose:
            with console.status(f"[cyan]{message}...[/cyan]", spinner="dots") as status:
                yield status
        else:
            yield None

    def log(self, message: str, style: str = "dim"):
        """Log a message to the console."""
        if self.verbose:
            console.print(f"[{style}]{message}[/{style}]")

    def log_info(self, message: str):
        """Log an info message with icon."""
        if self.verbose:
            console.print(f"[blue]ℹ[/blue] {message}")

    def log_success(self, message: str):
        """Log a success message with icon."""
        console.print(f"[green]✓[/green] {message}")

    def log_error(self, message: str):
        """Log an error message with icon."""
        console.print(f"[red]✗[/red] {message}")

    def log_model(self, action: str):
        """Log what the model is doing."""
        if self.verbose:
            agent_name = self.get_config().name
            console.print(f"[yellow]⚡[/yellow] [bold]{agent_name}[/bold] → {action} [dim]({self.model_name})[/dim]")

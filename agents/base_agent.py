"""
Base Agent Architecture

Defines the core agent interface and configuration structure.
Each agent is self-describing with:
- Unique ID
- Intent patterns for routing
- Capability description
- Example queries
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from langchain_openai import ChatOpenAI
from rich.console import Console
import os
from dotenv import load_dotenv

load_dotenv()
console = Console()


@dataclass
class AgentConfig:
    """
    Configuration for an agent.
    
    This makes agents self-describing and allows the dispatcher
    to route based on patterns without hard-coding logic.
    """
    agent_id: str                    # Unique identifier (e.g., "movie_librarian")
    name: str                        # Human-readable name (e.g., "Movie Librarian")
    description: str                 # What this agent does
    patterns: list[str]              # Routing patterns (e.g., ["add:", "ingest:"])
    keywords: list[str]              # Keyword triggers (e.g., ["add", "store"])
    capabilities: list[str]          # What it can do
    example_queries: list[str]       # Example usage
    requires_llm: bool = True        # Whether it needs LLM


class BaseAgent(ABC):
    """
    Base class for all agents.
    
    Each agent must:
    1. Define its configuration via get_config()
    2. Implement process() to handle requests
    """
    
    def __init__(self, model: str | None = None):
        """
        Initialize the agent.
        
        Args:
            model: Optional LLM model override
        """
        self.model_name = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b:free")
        self.llm = None
        
        # Initialize LLM if agent requires it
        if self.get_config().requires_llm:
            self.llm = self._init_llm()
    
    def _init_llm(self) -> ChatOpenAI:
        """Initialize the LLM client."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        
        return ChatOpenAI(
            model=self.model_name,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
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
    
    # Logging helpers
    def log(self, message: str, style: str = "dim"):
        """Log a message to the console."""
        console.print(f"[{style}]{message}[/{style}]")
    
    def log_info(self, message: str):
        """Log an info message with icon."""
        console.print(f"[blue]ℹ[/blue] {message}")
    
    def log_success(self, message: str):
        """Log a success message with icon."""
        console.print(f"[green]✓[/green] {message}")
    
    def log_error(self, message: str):
        """Log an error message with icon."""
        console.print(f"[red]✗[/red] {message}")

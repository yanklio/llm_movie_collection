"""
Base Agent - Abstract base class for all agents.

Provides shared utilities and configuration.
"""

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console

load_dotenv()
console = Console()


class BaseAgent(ABC):
    """
    Abstract base for all agents in the system.
    
    Shared responsibilities:
    - LLM initialization
    - Configuration loading
    - Logging utilities
    """
    
    def __init__(self, model: str | None = None):
        """Initialize the agent with optional LLM."""
        self.model_name = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b:free")
        self.llm = self._init_llm() if self.requires_llm() else None
    
    def _init_llm(self) -> ChatOpenAI:
        """Initialize LLM via OpenRouter."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required for LLM-based agents")
        
        return ChatOpenAI(
            model=self.model_name,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,  # Lower temp for more consistent behavior
        )
    
    @abstractmethod
    def requires_llm(self) -> bool:
        """Return True if this agent needs an LLM."""
        pass
    
    def log(self, message: str, style: str = "dim"):
        """Log a message to the console."""
        console.print(f"[{style}]{message}[/{style}]")
    
    def log_success(self, message: str):
        """Log a success message."""
        console.print(f"[green]✓[/green] {message}")
    
    def log_error(self, message: str):
        """Log an error message."""
        console.print(f"[red]✗[/red] {message}")
    
    def log_info(self, message: str):
        """Log an info message."""
        console.print(f"[blue]ℹ[/blue] {message}")

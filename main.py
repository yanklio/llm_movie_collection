#!/usr/bin/env python3
"""
Movie RAG System - Unified CLI

Extensible multi-agent system with automatic routing.

Usage:
    # Automatic routing
    python main.py "Add Inception"
    python main.py "Find dark sci-fi movies"
    
    # Direct agent selection
    python main.py --agent movie_librarian "Christopher Nolan movies"
    python main.py --agent movie_critic "emotional Tom Hanks movie"
    
    # List available agents
    python main.py --list-agents
"""

import sys
import argparse
from rich.console import Console
from rich.table import Table

from agents.dispatcher import Dispatcher
from agents.registry import AgentRegistry

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Movie RAG System - Extensible Multi-Agent Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic routing
  python main.py "Add Inception"
  python main.py "Find dark sci-fi about dreams"
  
  # Direct agent selection (bypass router)
  python main.py --agent movie_librarian "Tom Hanks movies"
  python main.py --agent movie_critic "emotional drama"
  
  # List all available agents
  python main.py --list-agents
        """
    )
    
    parser.add_argument(
        "query",
        nargs="*",
        help="Your request"
    )
    
    parser.add_argument(
        "--agent",
        "-a",
        help=f"Directly select agent by ID (bypasses router). Available: {', '.join(AgentRegistry.get_all_agents().keys())}"
    )
    
    parser.add_argument(
        "--model",
        "-m",
        help="Override LLM model (default: from .env)"
    )
    
    parser.add_argument(
        "--list-agents",
        "-l",
        action="store_true",
        help="List all available agents and their capabilities"
    )
    
    args = parser.parse_args()
    
    # List agents
    if args.list_agents:
        list_agents()
        return
    
    # Validate query
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    query = " ".join(args.query)
    
    console.print(f"\n[bold cyan]Movie RAG System[/bold cyan]")
    console.print(f"[dim]Query: {query}[/dim]\n")
    
    # Direct agent selection or auto-routing
    if args.agent:
        # Direct selection
        agent_class = AgentRegistry.get_agent(args.agent)
        if not agent_class:
            console.print(f"[red]Error:[/red] Unknown agent '{args.agent}'")
            console.print(f"[yellow]Available agents:[/yellow] {', '.join(AgentRegistry.get_all_agents().keys())}")
            sys.exit(1)
        
        config = agent_class.get_config()
        console.print(f"[yellow]Agent:[/yellow] {config.name} (direct)")
        console.print("[dim]" + "="*60 + "[/dim]\n")
        
        run_agent(agent_class, query, args.model)
    else:
        # Auto-routing
        console.print("[yellow]Mode:[/yellow] Auto-routing via Dispatcher")
        console.print("[dim]" + "="*60 + "[/dim]\n")
        
        dispatcher = Dispatcher(model=args.model)
        routing = dispatcher.process(query)
        
        agent_id = routing["agent_id"]
        clean_query = routing["query"]
        
        console.print()
        
        agent_class = AgentRegistry.get_agent(agent_id)
        if not agent_class:
            console.print(f"[red]Error:[/red] Routing failed")
            sys.exit(1)
        
        run_agent(agent_class, clean_query, args.model)


def run_agent(agent_class, query: str, model: str | None = None):
    """Execute an agent with a query."""
    config = agent_class.get_config()
    agent = agent_class(model=model)
    
    result = agent.process(query)
    
    # Display results based on agent type
    if config.agent_id == "movie_librarian":
        display_librarian_results(result)
    elif config.agent_id == "movie_critic":
        display_critic_results(result)
    else:
        console.print(result)


def display_librarian_results(result: dict):
    """Display Librarian agent results."""
    successful = result.get("successful", [])
    failed = result.get("failed", [])
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  ✓ Added: {len(successful)} movies")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)} movies")
    
    if successful:
        console.print(f"\n[green]Successfully added:[/green]")
        for movie in successful:
            console.print(f"  • {movie.title} ({movie.year})")


def display_critic_results(result: dict):
    """Display Critic agent results."""
    response = result.get("response", "")
    console.print(response)
    console.print()


def list_agents():
    """List all available agents."""
    console.print(f"\n[bold cyan]Available Agents[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Example Queries")
    
    configs = AgentRegistry.get_configs()
    for agent_id, config in configs.items():
        examples = "\n".join(f"• {ex}" for ex in config.example_queries[:2])
        table.add_row(
            agent_id,
            config.name,
            config.description,
            examples
        )
    
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()

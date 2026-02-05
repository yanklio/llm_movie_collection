#!/usr/bin/env python3

import argparse
import sys

from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

from agents.dispatcher import Dispatcher
from agents.registry import AgentRegistry

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Movie RAG System - Orchestrated Multi-Agent Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic orchestration
  python main.py "Add Inception"          # MovieCollector → Librarian
  python main.py "Find dark sci-fi"       # Critic

  # Direct agent selection (bypass orchestration)
  python main.py --agent movie_librarian "Tom Hanks movies"
  python main.py --agent movie_critic "emotional drama"

  # List all available agents
  python main.py --list-agents
        """,
    )

    parser.add_argument("query", nargs="*", help="Your request")

    parser.add_argument(
        "--agent",
        "-a",
        help=f"Directly select agent by ID (bypasses orchestration). Available: {', '.join(AgentRegistry.get_all_agents().keys())}",
    )

    parser.add_argument("--model", "-m", help="Override LLM model (default: from .env)")

    parser.add_argument(
        "--list-agents",
        "-l",
        action="store_true",
        help="List all available agents and their capabilities",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose mode to see agent reasoning and tool calls",
    )

    parser.add_argument(
        "--chat",
        "-c",
        action="store_true",
        help="Start interactive chat mode",
    )

    args = parser.parse_args()

    if args.list_agents:
        list_agents()
        return

    if args.chat:
        run_chat_mode(args.model, args.verbose)
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    query = " ".join(args.query)

    console.print("\n[bold cyan]Movie RAG System[/bold cyan]")
    console.print(f"[dim]Query: {query}[/dim]\n")

    if args.agent:
        run_direct_agent(args.agent, query, args.model)
        return

    run_orchestrated(query, args.model, args.verbose)


def run_chat_mode(model: str | None = None, verbose: bool = False):
    """Run interactive chat session with history."""
    console.print("[bold cyan]Movie RAG System - Interactive Chat[/bold cyan]")
    console.print("[dim]Type 'exit' or 'quit' to end session[/dim]\n")
    
    dispatcher = Dispatcher(model=model, verbose=verbose)
    history = []
    
    while True:
        try:
            query = console.input("[bold green]You > [/bold green]")
            if not query.strip():
                continue
                
            if query.lower() in ("exit", "quit"):
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            history.append({"role": "user", "content": query})
            
            with console.status("[dim]Thinking...[/dim]"):
                result = dispatcher.process(query, chat_history=history)
            
            display_result(result)
            
            # Extract text response for history
            response_text = result.get("response") or result.get("message", "")
            if not response_text and result.get("items"):
                response_text = f"Found {len(result['items'])} items."
            
            if response_text:
                history.append({"role": "assistant", "content": response_text})
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


def run_direct_agent(agent_id: str, query: str, model: str | None = None):
    """Run a specific agent directly (bypass orchestration)."""
    agent_class = AgentRegistry.get_agent(agent_id)
    if not agent_class:
        console.print(f"[red]Error:[/red] Unknown agent '{agent_id}'")
        console.print(
            f"[yellow]Available agents:[/yellow] {', '.join(AgentRegistry.get_all_agents().keys())}"
        )
        sys.exit(1)

    config = agent_class.get_config()
    console.print(f"[yellow]Agent:[/yellow] {config.name} (direct)")
    console.print("[dim]" + "=" * 60 + "[/dim]\n")

    agent = agent_class(model=model)
    result = agent.process(query)
    display_result(result)


def run_orchestrated(query: str, model: str | None = None, verbose: bool = False):
    """Run query through the dispatcher for automatic orchestration."""
    console.print("[yellow]Mode:[/yellow] Orchestrated Workflow")
    console.print("[dim]" + "=" * 60 + "[/dim]\n")

    dispatcher = Dispatcher(model=model, verbose=verbose)
    result = dispatcher.process(query)

    console.print()
    display_result(result)


def display_result(result: dict):
    """Display any agent result in a clean format."""
    response_text = result.get("response", "")
    
    if response_text:
        console.print("[bold cyan]━━━ Result ━━━[/bold cyan]\n")
        console.print(response_text)
        console.print()
        
        stored = result.get("stored_count", 0)
        skipped = result.get("skipped_count", 0)
        if stored > 0 or skipped > 0:
            console.print(f"[dim]📦 Stored: {stored} | ⏭️ Skipped: {skipped}[/dim]")
    
    elif result.get("items"):
        items = result.get("items", [])
        console.print("[bold cyan]━━━ Found Movies ━━━[/bold cyan]\n")
        for movie in items:
            if hasattr(movie, 'title'):
                console.print(f"  [bold]{movie.title}[/bold] ({movie.year})")
                console.print(f"  [dim]{movie.genre} • {movie.director} • ⭐ {movie.imdb_rating}[/dim]\n")
            else:
                console.print(f"  [bold]{movie.get('Title')}[/bold] ({movie.get('Year')})")
    
    elif result.get("error"):
        console.print(f"[red]Error:[/red] {result.get('error')}")
    
    elif result.get("message"):
        console.print("[bold cyan]━━━ Result ━━━[/bold cyan]\n")
        console.print(result.get("message"))
    
    else:
        console.print("[bold cyan]━━━ Result ━━━[/bold cyan]\n")
        console.print(Pretty(result))


def list_agents():
    """List all available agents."""
    console.print("\n[bold cyan]Available Agents[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Example Queries")

    configs = AgentRegistry.get_configs()
    for agent_id, config in configs.items():
        examples = "\n".join(f"• {ex}" for ex in config.example_queries[:2])
        table.add_row(agent_id, config.name, config.description, examples)

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()


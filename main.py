#!/usr/bin/env python3

import argparse
import sys

from rich.console import Console
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

    args = parser.parse_args()

    if args.list_agents:
        list_agents()
        return

    # Validate query
    if not args.query:
        parser.print_help()
        sys.exit(1)

    query = " ".join(args.query)

    console.print("\n[bold cyan]Movie RAG System[/bold cyan]")
    console.print(f"[dim]Query: {query}[/dim]\n")

    if args.agent:
        agent_class = AgentRegistry.get_agent(args.agent)
        if not agent_class:
            console.print(f"[red]Error:[/red] Unknown agent '{args.agent}'")
            console.print(
                f"[yellow]Available agents:[/yellow] {', '.join(AgentRegistry.get_all_agents().keys())}"
            )
            sys.exit(1)

        config = agent_class.get_config()
        console.print(f"[yellow]Agent:[/yellow] {config.name} (direct)")
        console.print("[dim]" + "=" * 60 + "[/dim]\n")

        run_agent(agent_class, query, args.model)
    else:
        # Auto-orchestration via Dispatcher
        console.print("[yellow]Mode:[/yellow] Orchestrated Workflow")
        console.print("[dim]" + "=" * 60 + "[/dim]\n")

        dispatcher = Dispatcher(model=args.model, verbose=args.verbose)
        result = dispatcher.process(query)

        console.print()

        # Display results based on orchestration flow
        intent = result.get("intent", "unknown")

        if intent == "add_movie":
            # MovieCollector → Librarian workflow
            console.print(f"[cyan]Workflow:[/cyan] {result.get('workflow', 'unknown')}\n")
            storage_result = result.get("storage_result", {})
            display_librarian_results(storage_result)

        elif intent == "fetch_movie":
            # MovieCollector only
            console.print(f"[cyan]Workflow:[/cyan] {result.get('workflow', 'unknown')}\n")
            movie_data = result.get("movie_data", {})
            console.print("[bold]Movie Data:[/bold]")
            console.print(movie_data)
            console.print()

        elif intent == "query_movie":
            # Critic workflow
            console.print("[cyan]Workflow:[/cyan] Critic\n")
            display_critic_results(result)

        elif intent == "check_movie":
            # Librarian check workflow
            console.print("[cyan]Workflow:[/cyan] Librarian (Check)\n")
            exists = result.get("exists", False)

            if exists:
                movie = result.get("movie", {})
                console.print(
                    f"[green]✓ Yes![/green] {movie.get('title', 'Unknown')} ({movie.get('year', 'N/A')}) is in your watchlist"
                )
                console.print(
                    f"[dim]Genre: {movie.get('genre', 'N/A')} | Rating: {movie.get('rating', 'N/A')}[/dim]"
                )
            else:
                message = result.get("message", "Movie not found")
                console.print(f"[yellow]✗ No.[/yellow] {message}")

        elif intent == "delete_movie":
            # Librarian delete workflow
            console.print("[cyan]Workflow:[/cyan] Librarian (Delete)\n")
            success = result.get("success", False)
            message = result.get("message", "Unknown")

            if success:
                movie = result.get("movie", {})
                console.print(
                    f"[green]✓ Deleted:[/green] {movie.get('title', 'Unknown')} ({movie.get('year', 'N/A')})"
                )
            else:
                console.print(f"[red]✗ Failed:[/red] {message}")

        else:
            console.print(f"[red]Unknown intent:[/red] {intent}")
            console.print(result)


def run_agent(agent_class, query: str, model: str | None = None):
    """Execute an agent directly (no orchestration)."""
    config = agent_class.get_config()
    agent = agent_class(model=model)

    result = agent.process(query)

    # Display results based on agent type
    if config.agent_id == "movie_librarian":
        display_librarian_results(result)
    elif config.agent_id == "movie_critic":
        display_critic_results(result)
    elif config.agent_id == "movie_collector":
        console.print("[bold]Collector Result:[/bold]")
        console.print(result)
        console.print()
    else:
        console.print(result)


def display_librarian_results(result: dict):
    """Display Librarian agent results."""
    successful = result.get("successful", [])
    failed = result.get("failed", [])

    console.print("[bold]Storage Summary:[/bold]")
    console.print(f"  ✓ Added: {len(successful)} movies")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)} movies")

    if successful:
        console.print("\n[green]Successfully added:[/green]")
        for movie in successful:
            console.print(f"  • {movie.title} ({movie.year})")

    console.print()


def display_critic_results(result: dict):
    """Display Critic agent results."""
    response = result.get("response", "")
    if response:
        console.print(response)
    else:
        console.print("[yellow]No recommendations found[/yellow]")
    console.print()


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

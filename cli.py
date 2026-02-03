#!/usr/bin/env python3
"""
Movie RAG System - Unified CLI

Automatically routes requests to the appropriate agent:
- Librarian: For adding movies
- Critic: For querying/recommendations

Usage:
    # Automatic routing
    python movie_cli.py "Add Inception"
    python movie_cli.py "Find dark sci-fi movies"
    
    # Direct agent selection
    python movie_cli.py --agent librarian "Christopher Nolan movies"
    python movie_cli.py --agent critic "emotional Tom Hanks movie"
"""

import sys
import argparse
from agents.dispatcher import Dispatcher, Intent
from agents.librarian import Librarian
from agents.critic import Critic
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Movie RAG System - Add movies or get recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic routing
  python movie_cli.py "Add Inception"
  python movie_cli.py "Find dark sci-fi about dreams"
  
  # Direct agent selection (bypass router)
  python movie_cli.py --agent librarian "Tom Hanks movies"
  python movie_cli.py --agent critic "emotional drama"
  
  # Explicit commands
  python movie_cli.py "add: Christopher Nolan films"
  python movie_cli.py "find: action movies from the 90s"
        """
    )
    
    parser.add_argument(
        "query",
        nargs="+",
        help="Your request (e.g., 'Add Inception' or 'Find dark sci-fi')"
    )
    
    parser.add_argument(
        "--agent",
        "-a",
        choices=["librarian", "critic", "auto"],
        default="auto",
        help="Directly select agent (bypasses router). Default: auto-detect"
    )
    
    parser.add_argument(
        "--model",
        "-m",
        help="Override LLM model (default: from .env)"
    )
    
    args = parser.parse_args()
    
    # Combine query parts
    query = " ".join(args.query)
    
    console.print(f"\n[bold cyan]Movie RAG System[/bold cyan]")
    console.print(f"[dim]Query: {query}[/dim]\n")
    
    # Direct agent selection or auto-routing
    if args.agent == "librarian":
        console.print("[yellow]Agent:[/yellow] Librarian (direct)")
        console.print("[dim]" + "="*60 + "[/dim]\n")
        run_librarian(query, args.model)
        
    elif args.agent == "critic":
        console.print("[yellow]Agent:[/yellow] Critic (direct)")
        console.print("[dim]" + "="*60 + "[/dim]\n")
        run_critic(query, args.model)
        
    else:
        # Auto-routing via Dispatcher
        console.print("[yellow]Agent:[/yellow] Auto-routing via Dispatcher")
        console.print("[dim]" + "="*60 + "[/dim]\n")
        
        dispatcher = Dispatcher(model=args.model)
        intent = dispatcher.route(query)
        
        # Extract clean query (remove prefixes)
        clean_query = dispatcher.extract_query(query)
        
        console.print()
        
        if intent == Intent.ADD_MOVIE:
            run_librarian(clean_query, args.model)
        elif intent == Intent.QUERY_MOVIES:
            run_critic(clean_query, args.model)
        else:
            console.print("[red]⚠️  Unable to determine intent[/red]")
            sys.exit(1)


def run_librarian(query: str, model: str | None = None):
    """Run Librarian agent."""
    librarian = Librarian(model=model)
    
    successful, failed = librarian.process_request(query)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  ✓ Added: {len(successful)} movies")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)} movies")
    
    if successful:
        console.print(f"\n[green]Successfully added:[/green]")
        for movie in successful:
            console.print(f"  • {movie.title} ({movie.year})")


def run_critic(query: str, model: str | None = None):
    """Run Critic agent."""
    critic = Critic(model=model)
    
    response = critic.query(query)
    console.print(response)
    console.print()


if __name__ == "__main__":
    main()

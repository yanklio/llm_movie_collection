#!/usr/bin/env python3
"""
Movie RAG System - CLI Interface

Usage:
    python main.py add "Movie Title"
    python main.py query "dark sci-fi about dreams"
    python main.py list
    python main.py chat
"""

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


def cmd_add(args):
    """Add movies to knowledge base."""
    from ingest.ingestor import Ingestor

    ingestor = Ingestor()
    for movie in args.movies:
        try:
            ingestor.ingest(movie, force_update=args.force)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


def cmd_query(args):
    """Query for movie recommendations."""
    from rag.pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline(model=args.model)
        response = pipeline.query(args.query, top_k=args.top_k)

        console.print()
        console.print(
            Panel(
                response,
                title="[bold blue]Recommendations[/bold blue]",
                border_style="blue",
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def cmd_list(args):
    """List all movies."""
    from storage.vector_store import VectorStore

    store = VectorStore()
    movies = store.get_all_movies()

    if not movies:
        console.print("[yellow]No movies in database.[/yellow]")
        return

    console.print(f"\n[bold]Movies ({len(movies)}):[/bold]")
    for m in movies:
        meta = m["metadata"]
        console.print(
            f"  • {meta.get('title')} ({meta.get('year')}) - {meta.get('genre', 'N/A')[:40]}"
        )


def cmd_stats(args):
    """Show database stats."""
    from storage.vector_store import VectorStore

    store = VectorStore()
    count = store.count()
    movies = store.get_all_movies()

    if movies:
        avg_len = sum(len(m["document"]) for m in movies) / len(movies)
    else:
        avg_len = 0

    console.print("\n[bold]Database Stats:[/bold]")
    console.print(f"  Movies: {count}")
    console.print(f"  Avg doc length: {avg_len:.0f} chars")


def cmd_chat(args):
    """Interactive chat mode."""
    from rag.pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline(model=args.model)
        pipeline.chat()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Movie RAG System")
    subparsers = parser.add_subparsers(dest="command")

    # add
    add_p = subparsers.add_parser("add", help="Add movies")
    add_p.add_argument("movies", nargs="+", help="Movie titles or IMDb URLs")
    add_p.add_argument("-f", "--force", action="store_true", help="Update existing")
    add_p.set_defaults(func=cmd_add)

    # query
    query_p = subparsers.add_parser("query", help="Query for recommendations")
    query_p.add_argument("query", help="What you're looking for")
    query_p.add_argument("-k", "--top-k", type=int, default=5, help="Results to use")
    query_p.add_argument("-m", "--model", help="LLM model override")
    query_p.set_defaults(func=cmd_query)

    # list
    list_p = subparsers.add_parser("list", help="List movies")
    list_p.set_defaults(func=cmd_list)

    # stats
    stats_p = subparsers.add_parser("stats", help="Show stats")
    stats_p.set_defaults(func=cmd_stats)

    # chat
    chat_p = subparsers.add_parser("chat", help="Interactive chat")
    chat_p.add_argument("-m", "--model", help="LLM model override")
    chat_p.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

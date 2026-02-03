#!/usr/bin/env python3
"""
Autonomous Librarian CLI

Usage:
    python librarian_cli.py "Add Inception"
    python librarian_cli.py "Add all Christopher Nolan movies"
    python librarian_cli.py "Find sci-fi movies from 1999"
"""

import sys
from agents.librarian import Librarian
from rich.console import Console

console = Console()

def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/red] python librarian_cli.py \"<your request>\"")
        console.print("\n[yellow]Examples:[/yellow]")
        console.print('  python librarian_cli.py "Add Inception"')
        console.print('  python librarian_cli.py "Add Christopher Nolan movies"')
        console.print('  python librarian_cli.py "Find Tom Hanks movies"')
        sys.exit(1)
    
    # Get the request from command line
    request = " ".join(sys.argv[1:])
    
    # Initialize Librarian
    console.print(f"\n[bold cyan]Librarian Agent[/bold cyan]")
    console.print(f"[dim]Model: openai/gpt-oss-20b:free[/dim]\n")
    
    librarian = Librarian()
    
    # Process the request
    successful, failed = librarian.process_request(request)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  ✓ Added: {len(successful)} movies")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)} movies")
    
    if successful:
        console.print(f"\n[green]Successfully added:[/green]")
        for movie in successful:
            console.print(f"  • {movie.title} ({movie.year})")

if __name__ == "__main__":
    main()

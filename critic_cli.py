#!/usr/bin/env python3
"""
Critic Agent CLI - Query for movie recommendations

Usage:
    python critic_cli.py "dark sci-fi about dreams"
    python critic_cli.py "I want something emotional starring Tom Hanks"
    python critic_cli.py "Find me Christopher Nolan movies"
"""

import sys
from agents.critic import Critic
from rich.console import Console

console = Console()

def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/red] python critic_cli.py \"<your query>\"")
        console.print("\n[yellow]Examples:[/yellow]")
        console.print('  python critic_cli.py "dark sci-fi about reality and dreams"')
        console.print('  python critic_cli.py "I want something emotional starring Tom Hanks"')
        console.print('  python critic_cli.py "Find me Christopher Nolan movies"')
        console.print('  python critic_cli.py "action movie from the 90s"')
        sys.exit(1)
    
    # Get the query from command line
    query = " ".join(sys.argv[1:])
    
    # Initialize Critic
    console.print(f"\n[bold cyan]Critic Agent[/bold cyan]")
    console.print(f"[dim]Model: openai/gpt-oss-20b:free[/dim]\n")
    
    critic = Critic()
    
    # Process the query
    console.print(f"[yellow]Query:[/yellow] {query}")
    console.print("[dim]" + "="*60 + "[/dim]\n")
    
    response = critic.query(query)
    
    console.print(response)
    console.print()

if __name__ == "__main__":
    main()

"""
RAGPipeline - Simple RAG without tool calling.

Searches knowledge base and passes context to LLM.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from storage.vector_store import VectorStore
from tools.movie_api import MovieAPITool

load_dotenv()
console = Console()

DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

SYSTEM_PROMPT = """You are a helpful movie recommendation assistant.

Based on the movies provided in the context, recommend ones that match the user's request.
Consider mood, genre, themes, and vibes when making recommendations.
Explain why each recommendation fits what the user is looking for.

If no movies in the context match well, say so honestly."""


class RAGPipeline:
    """
    Simple RAG pipeline - searches knowledge base and provides context to LLM.
    """

    def __init__(
        self,
        model: str | None = None,
        movie_api: MovieAPITool | None = None,
        vector_store: VectorStore | None = None,
    ):
        """Initialize the RAG pipeline."""
        self.movie_api = movie_api or MovieAPITool()
        self.vector_store = vector_store or VectorStore()

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required")

        self.model_name = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)

        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
        )

    def query(self, user_input: str, top_k: int = 5) -> str:
        """Process user query through RAG pipeline."""
        console.print("[dim]Searching knowledge base...[/dim]")

        # Search for relevant movies
        results = self.vector_store.search(user_input, top_k=top_k)

        if not results:
            return "No movies in knowledge base. Please add some movies first."

        # Build context from results
        context = self._build_context(results)

        console.print(f"[dim]Found {len(results)} relevant movies. Generating response...[/dim]")

        # Generate response
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"""Movies in database:

{context}

User request: {user_input}

Recommend matching movies and explain why they fit."""
            ),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _build_context(self, results: list[dict]) -> str:
        """Build context string from search results."""
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            parts.append(
                f"{i}. {meta.get('title')} ({meta.get('year')})\n"
                f"   Genre: {meta.get('genre')}\n"
                f"   Director: {meta.get('director')}\n"
                f"   Rating: {meta.get('imdb_rating')}/10\n"
                f"   Plot: {meta.get('plot', 'N/A')[:200]}..."
            )
        return "\n\n".join(parts)

    def add_movie(self, query: str) -> str:
        """Add a movie to the knowledge base."""
        try:
            movie = self.movie_api.search(query)
            added = self.vector_store.add_movie(movie.imdb_id, movie.to_document(), movie.to_dict())
            if added:
                return f"Added '{movie.title}' ({movie.year})"
            return f"'{movie.title}' already exists"
        except Exception as e:
            return f"Error: {e}"

    def list_movies(self) -> str:
        """List all movies in knowledge base."""
        movies = self.vector_store.get_all_movies()
        if not movies:
            return "Knowledge base is empty."

        lines = [f"Knowledge base ({len(movies)} movies):"]
        for m in movies:
            meta = m["metadata"]
            lines.append(f"  - {meta.get('title')} ({meta.get('year')})")
        return "\n".join(lines)

    def chat(self) -> None:
        """Interactive chat loop."""
        console.print("[bold blue]Movie Assistant[/bold blue]")
        console.print("Commands: 'add <title>', 'list', 'quit'\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if user_input.lower().startswith("add "):
                    result = self.add_movie(user_input[4:])
                    console.print(f"[green]{result}[/green]\n")
                    continue
                if user_input.lower() == "list":
                    console.print(self.list_movies() + "\n")
                    continue

                response = self.query(user_input)
                console.print(f"\n[green]Assistant:[/green] {response}\n")

            except KeyboardInterrupt:
                break

        console.print("\n[dim]Goodbye![/dim]")

"""
Ingestor - Orchestrates the movie ingestion pipeline.

Handles the write path: fetching movie data from external APIs,
and storing it in the vector database.
"""

from rich.console import Console

from storage.vector_store import VectorStore
from tools.movie_api import MovieAPITool
from tools.movie_info import MovieInfo

console = Console()


class Ingestor:
    """
    Orchestrates movie data ingestion.

    Flow:
    1. Parse input (title, IMDb ID, or URL)
    2. Fetch data from OMDb API
    3. Store in VectorStore
    """

    def __init__(
        self,
        movie_api: MovieAPITool | None = None,
        vector_store: VectorStore | None = None,
    ):
        """Initialize the Ingestor."""
        self.movie_api = movie_api or MovieAPITool()
        self.vector_store = vector_store or VectorStore()

    def ingest(self, query: str, force_update: bool = False) -> MovieInfo:
        """
        Ingest a movie into the vector store.

        Args:
            query: Movie title, IMDb ID, or IMDb URL.
            force_update: If True, update existing movie data.

        Returns:
            MovieInfo object of the ingested movie.
        """
        console.print(f"[dim]Fetching movie data for: {query}[/dim]")

        # Fetch movie data from API
        movie_info = self.movie_api.search(query)

        console.print(f"[green]Found:[/green] {movie_info.title} ({movie_info.year})")

        # Convert to document format for embedding
        document = movie_info.to_document()
        metadata = movie_info.to_dict()

        # Store in vector database
        if force_update:
            self.vector_store.update_movie(
                imdb_id=movie_info.imdb_id,
                document=document,
                metadata=metadata,
            )
            console.print(f"[blue]Updated:[/blue] {movie_info.title}")
        else:
            added = self.vector_store.add_movie(
                imdb_id=movie_info.imdb_id,
                document=document,
                metadata=metadata,
            )
            if not added:
                console.print(f"[yellow]Already exists:[/yellow] {movie_info.title}")
                raise ValueError(
                    f"Movie '{movie_info.title}' already in database. Use --force to update."
                )
            console.print(f"[green]Added:[/green] {movie_info.title}")

        return movie_info

    def ingest_batch(
        self,
        queries: list[str],
        force_update: bool = False,
    ) -> tuple[list[MovieInfo], list[str]]:
        """
        Ingest multiple movies.

        Returns:
            Tuple of (successful MovieInfo list, failed query list).
        """
        successful: list[MovieInfo] = []
        failed: list[str] = []

        for query in queries:
            try:
                movie = self.ingest(query, force_update=force_update)
                successful.append(movie)
            except Exception as e:
                console.print(f"[red]Failed:[/red] {query} - {e}")
                failed.append(query)

        console.print(f"\n[bold]Summary:[/bold] {len(successful)} added, {len(failed)} failed")
        return successful, failed

    def remove(self, imdb_id: str) -> bool:
        """Remove a movie from the vector store."""
        removed = self.vector_store.delete_movie(imdb_id)
        if removed:
            console.print(f"[green]Removed:[/green] {imdb_id}")
        else:
            console.print(f"[yellow]Not found:[/yellow] {imdb_id}")
        return removed

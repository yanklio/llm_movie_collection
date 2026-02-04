"""
Librarian Agent - Pure storage specialist.

Responsibilities:
- Store movie data in vector database
- NO API calls - only receives data from Dispatcher
- Manages movie metadata and summaries
"""

from typing import List

from agents.base_agent import BaseAgent, AgentConfig
from tools.movies.movie_info import MovieInfo
from storage.vector_store import VectorStore


class Librarian(BaseAgent):
    """
    Storage Agent - Stores movie data provided by Dispatcher.
    
    Does NOT fetch data - only receives and stores.
    """
    
    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return agent configuration."""
        return AgentConfig(
            agent_id="movie_librarian",
            name="Movie Librarian",
            description="Storage specialist. Receives movie data from Dispatcher and stores in vector database.",
            patterns=["add:", "add ", "ingest:", "ingest ", "store:", "store "],
            keywords=["add", "ingest", "store", "save", "import"],
            capabilities=[
                "Store movie metadata",
                "Create searchable summaries",
                "Manage vector database"
            ],
            example_queries=[
                "Add Inception",
                "Store Christopher Nolan films"
            ],
            requires_llm=False  # No LLM needed - just storage
        )
    
    def __init__(
        self,
        model: str | None = None,
        verbose: bool = False,
        vector_store: VectorStore | None = None,
    ):
        """Initialize the Librarian."""
        super().__init__(model, verbose)
        self.vector_store = vector_store or VectorStore()
    
    def process(self, query: str) -> dict:
        """
        Process a storage request.
        
        NOTE: This is kept for compatibility but should not be used.
        Use store_movies() directly instead.
        """
        return {
            "message": "Use store_movies() with movie data",
            "agent": self.get_config().agent_id
        }
    
    def store_movies(self, movies: List) -> dict:
        """
        Store a list of movies in the vector database.
        
        Args:
            movies: List of MovieInfo objects or dicts from TMDB API
            
        Returns:
            Dictionary with 'successful' and 'failed' lists
        """
        successful = []
        failed = []
        
        for movie in movies:
            try:
                # If already a MovieInfo object, use directly
                if isinstance(movie, MovieInfo):
                    movie_info = movie
                else:
                    # Convert dict to MovieInfo
                    movie_info = self._dict_to_movie_info(movie)
                
                # Create summary
                summary = self._create_summary(movie_info)
                
                # Store in vector database
                self._store_in_db(movie_info, summary)
                
                successful.append(movie_info)
                self.log(f"✓ Stored: {movie_info.title} ({movie_info.year})")
                
            except Exception as e:
                if isinstance(movie, MovieInfo):
                    movie_title = f"{movie.title} ({movie.year})"
                else:
                    movie_title = movie.get('Title', 'Unknown') if hasattr(movie, 'get') else 'Unknown'
                self.log_error(f"✗ Failed to store {movie_title}: {e}")
                failed.append(movie_title)
        
        return {
            "successful": successful,
            "failed": failed,
            "agent": self.get_config().agent_id
        }
    
    def _dict_to_movie_info(self, movie_dict: dict) -> MovieInfo:
        """Convert TMDB API dict to MovieInfo object."""
        return MovieInfo(
            imdb_id=movie_dict.get('imdbID', ''),
            title=movie_dict.get('Title', ''),
            year=movie_dict.get('Year', ''),
            genre=movie_dict.get('Genre', ''),
            director=movie_dict.get('Director', ''),
            actors=movie_dict.get('Actors', ''),
            plot=movie_dict.get('Plot', ''),
            runtime=movie_dict.get('Runtime', ''),
            imdb_rating=movie_dict.get('imdbRating', 'N/A'),
            poster_url=movie_dict.get('Poster', '')
        )
    
    def _create_summary(self, movie: MovieInfo) -> str:
        """Create a searchable summary for the movie."""
        return f"""Title: {movie.title} ({movie.year})
Genre: {movie.genre}
Director: {movie.director}
Cast: {movie.actors}
Rating: {movie.imdb_rating}/10

Plot: {movie.plot}

Runtime: {movie.runtime}"""
    
    def _store_in_db(self, movie: MovieInfo, summary: str):
        """Store movie in vector database."""
        self.vector_store.add_movie(
            imdb_id=movie.imdb_id,
            document=summary,
            metadata={
                "title": movie.title,
                "year": movie.year,
                "genre": movie.genre,
                "director": movie.director,
                "actors": movie.actors,
                "rating": movie.imdb_rating
            }
        )

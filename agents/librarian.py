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
                "Manage vector database",
                "Check if movie exists in watchlist",
                "Delete movies from watchlist"
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
        
        Handles:
        - Check queries: "Do I have Inception?", "Is The Matrix in my watchlist?"
        - Delete queries: "Remove Inception", "Delete The Matrix"
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with operation result
        """
        query_lower = query.lower()
        
        # Check if it's a "check" query
        check_patterns = ["do i have", "is ", " in ", "already have", "watchlist"]
        if any(pattern in query_lower for pattern in check_patterns):
            # Extract movie title (simple heuristic)
            title = query
            for pattern in ["do i have ", "is ", " in my watchlist", " in watchlist", "already have "]:
                title = title.replace(pattern, "").strip()
            title = title.rstrip("?").strip()
            
            return self.check_movie(title)
        
        # Check if it's a "delete" query
        delete_patterns = ["remove", "delete", "drop"]
        if any(pattern in query_lower for pattern in delete_patterns):
            # Extract movie title
            title = query
            for pattern in ["remove ", "delete ", "drop "]:
                title = title.replace(pattern, "").strip()
            
            return self.delete_movie(title)
        
        return {
            "message": "Use store_movies() for adding, check_movie() for checking, delete_movie() for removing",
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
    
    def check_movie(self, title: str) -> dict:
        """
        Check if a movie is already in the watchlist.
        
        Args:
            title: Movie title to check
            
        Returns:
            Dictionary with existence status and movie info if found
        """
        self.log(f"Checking for: {title}")
        
        # Search in vector store
        results = self.vector_store.search(title, top_k=1)
        
        if results and len(results) > 0:
            movie = results[0]
            metadata = movie.get("metadata", {})
            self.log_success(f"✓ Found: {metadata.get('title', 'Unknown')} ({metadata.get('year', 'N/A')})")
            
            return {
                "exists": True,
                "movie": {
                    "title": metadata.get("title", "Unknown"),
                    "year": metadata.get("year", "N/A"),
                    "genre": metadata.get("genre", "N/A"),
                    "rating": metadata.get("rating", "N/A")
                },
                "agent": self.get_config().agent_id
            }
        else:
            self.log(f"✗ Not found: {title}")
            return {
                "exists": False,
                "message": f"'{title}' is not in your watchlist",
                "agent": self.get_config().agent_id
            }
    
    def delete_movie(self, title: str) -> dict:
        """
        Delete a movie from the watchlist.
        
        Args:
            title: Movie title to delete
            
        Returns:
            Dictionary with deletion status
        """
        self.log(f"Attempting to delete: {title}")
        
        # First, find the movie to get its IMDb ID
        results = self.vector_store.search(title, top_k=1)
        
        if not results or len(results) == 0:
            self.log_error(f"✗ Movie not found: {title}")
            return {
                "success": False,
                "message": f"'{title}' is not in your watchlist",
                "agent": self.get_config().agent_id
            }
        
        # Get the movie's ID (should be IMDb ID)
        movie = results[0]
        imdb_id = movie.get("id")
        metadata = movie.get("metadata", {})
        movie_title = metadata.get("title", title)
        
        # Delete from vector store
        success = self.vector_store.delete_movie(imdb_id)
        
        if success:
            self.log_success(f"✓ Deleted: {movie_title}")
            return {
                "success": True,
                "message": f"Removed '{movie_title}' from your watchlist",
                "movie": {
                    "title": movie_title,
                    "year": metadata.get("year", "N/A")
                },
                "agent": self.get_config().agent_id
            }
        else:
            self.log_error(f"✗ Failed to delete: {movie_title}")
            return {
                "success": False,
                "message": f"Failed to remove '{movie_title}'",
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

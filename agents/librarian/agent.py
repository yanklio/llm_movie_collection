import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.base_agent import AgentConfig, BaseAgent
from agents.librarian.prompts import (
    get_movie_summary_prompt,
    get_query_understanding_prompt,
    get_storage_confirmation_prompt,
)
from storage.vector_store import VectorStore
from tools.movies.movie_info import MovieInfo


class Librarian(BaseAgent):
    """
    Storage Agent - LLM-enhanced movie data storage and management.

    Uses LLMs to:
    - Generate rich, searchable movie summaries
    - Understand natural language queries about the watchlist
    - Provide conversational confirmations and responses
    """

    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return agent configuration."""
        return AgentConfig(
            agent_id="movie_librarian",
            name="Movie Librarian",
            description="LLM-enhanced storage specialist. Creates rich summaries and provides conversational watchlist management.",
            patterns=["add:", "add ", "store:", "store ", "save:", "save "],
            capabilities=[
                "LLM-generated movie summaries",
                "Natural language query understanding",
                "Conversational confirmations",
                "Smart watchlist management",
                "Semantic movie storage",
            ],
            example_queries=[
                "Add Inception to my watchlist",
                "Do I have The Matrix?",
                "Remove Pulp Fiction",
                "Tell me about the movies I have",
            ],
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

    def get_tools(self) -> List:
        """Return list of available tools for this agent."""
        return [
            self.check_movie_in_watchlist,
            self.delete_movie_from_watchlist,
            self.search_watchlist_movies,
            self.get_watchlist_statistics,
            self.get_movie_details,
        ]

    @tool
    def check_movie_in_watchlist(self, title: str) -> dict:
        """Check if a movie is in the watchlist."""
        self.log(f"Checking for: {title}")
        results = self.vector_store.search(title, top_k=1)

        if results and len(results) > 0:
            movie = results[0]
            metadata = movie.get("metadata", {})
            self.log_success(
                f"✓ Found: {metadata.get('title', 'Unknown')} ({metadata.get('year', 'N/A')})"
            )

            return {
                "exists": True,
                "movie": {
                    "title": metadata.get("title", "Unknown"),
                    "year": metadata.get("year", "N/A"),
                    "genre": metadata.get("genre", "N/A"),
                    "rating": metadata.get("rating", "N/A"),
                },
                "message": f"Yes! {metadata.get('title', 'Unknown')} ({metadata.get('year', 'N/A')}) is in your watchlist.",
            }
        else:
            self.log(f"✗ Not found: {title}")
            return {
                "exists": False,
                "message": f"'{title}' is not in your watchlist yet. Would you like to add it?",
            }

    @tool
    def delete_movie_from_watchlist(self, title: str) -> dict:
        """Delete a movie from the watchlist."""
        self.log(f"Attempting to delete: {title}")
        results = self.vector_store.search(title, top_k=1)

        if not results or len(results) == 0:
            self.log_error(f"✗ Movie not found: {title}")
            return {
                "success": False,
                "message": f"'{title}' is not in your watchlist, so nothing to remove.",
            }

        movie = results[0]
        imdb_id = movie.get("id")
        metadata = movie.get("metadata", {})
        movie_title = metadata.get("title", title)

        success = self.vector_store.delete_movie(imdb_id)

        if success:
            self.log_success(f"✓ Deleted: {movie_title}")
            return {
                "success": True,
                "message": f"Successfully removed '{movie_title}' from your watchlist.",
                "movie": {"title": movie_title, "year": metadata.get("year", "N/A")},
            }
        else:
            self.log_error(f"✗ Failed to delete: {movie_title}")
            return {
                "success": False,
                "message": f"Failed to remove '{movie_title}'. Please try again.",
            }

    @tool
    def search_watchlist_movies(self, query: str, limit: int = 5) -> dict:
        """Search for movies in the watchlist by themes, mood, or description."""
        self.log(f"Searching watchlist for: {query}")

        try:
            results = self.vector_store.search(query, top_k=limit)

            if not results:
                return {
                    "movies": [],
                    "count": 0,
                    "message": "No movies found matching your search.",
                }

            formatted_results = []
            for result in results:
                metadata = result.get("metadata", {})
                formatted_results.append(
                    {
                        "title": metadata.get("title", "Unknown"),
                        "year": metadata.get("year", "N/A"),
                        "genre": metadata.get("genre", "N/A"),
                        "rating": metadata.get("rating", "N/A"),
                        "similarity": result.get("distance", 0),
                    }
                )

            return {
                "movies": formatted_results,
                "count": len(formatted_results),
                "message": f"Found {len(formatted_results)} movies matching '{query}'",
            }

        except Exception as e:
            self.log_error(f"Watchlist search failed: {e}")
            return {"error": str(e)}

    @tool
    def get_watchlist_statistics(self) -> dict:
        """Get statistics about the movie watchlist."""
        try:
            total_movies = (
                self.vector_store.get_total_count()
                if hasattr(self.vector_store, "get_total_count")
                else 0
            )

            return {
                "total_movies": total_movies,
                "message": f"Your watchlist contains {total_movies} movies.",
            }
        except Exception as e:
            self.log_error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}

    @tool
    def get_movie_details(self, title: str) -> dict:
        """Get detailed information about a specific movie in the watchlist."""
        results = self.vector_store.search(title, top_k=1)

        if not results:
            return {
                "found": False,
                "message": f"'{title}' is not in your watchlist.",
            }

        movie = results[0]
        metadata = movie.get("metadata", {})
        document = movie.get("document", "")

        return {
            "found": True,
            "movie": {
                "title": metadata.get("title", "Unknown"),
                "year": metadata.get("year", "N/A"),
                "genre": metadata.get("genre", "N/A"),
                "director": metadata.get("director", "Unknown"),
                "actors": metadata.get("actors", "Unknown"),
                "rating": metadata.get("rating", "N/A"),
            },
            "summary": document,
        }

    def process(self, query: str) -> dict:
        """Process a natural language storage/management request."""
        self.log(f"Understanding query with LLM: {query}")

        try:
            # Step 1: LLM-powered query understanding
            intent = self._understand_query_with_llm(query)
            if not intent:
                return self._fallback_processing(query)

            operation = intent.get("operation", "unknown")
            movie_title = intent.get("movie_title")
            confidence = intent.get("confidence", 0.5)

            self.log(
                f"LLM Intent: {operation} operation for '{movie_title}' (confidence: {confidence:.2f})"
            )

            # Step 2: Execute appropriate operation
            if operation == "check":
                return self.check_movie_in_watchlist(movie_title)
            elif operation == "delete":
                return self.delete_movie_from_watchlist(movie_title)
            elif operation == "search":
                return self.search_watchlist_movies(query)
            elif operation == "info":
                return self.get_movie_details(movie_title)
            else:
                return {
                    "message": "I can help you add, check, delete, or search for movies in your watchlist.",
                    "suggested_actions": [
                        "Add a movie: 'Add Inception'",
                        "Check for a movie: 'Do I have The Matrix?'",
                        "Remove a movie: 'Delete Pulp Fiction'",
                        "Search collection: 'Show me sci-fi movies'",
                    ],
                    "agent": self.get_config().agent_id,
                }

        except Exception as e:
            self.log_error(f"Query processing failed: {e}")
            return {"error": str(e), "agent": self.get_config().agent_id}

    def store_movies(self, movies: List) -> dict:
        """Store a list of movies with LLM-generated summaries."""
        successful = []
        failed = []

        for movie in movies:
            try:
                movie_info = self._convert_to_movie_info(movie)
                summary = self._generate_rich_summary(movie_info)
                self._store_in_db(movie_info, summary)

                successful.append(movie_info)
                self.log(f"✓ Stored: {movie_info.title} ({movie_info.year})")

            except Exception as e:
                movie_title = self._get_movie_title(movie)
                self.log_error(f"✗ Failed to store {movie_title}: {e}")
                failed.append(movie_title)

        confirmation = self._generate_storage_confirmation(successful, failed)

        return {
            "successful": successful,
            "failed": failed,
            "confirmation": confirmation,
            "agent": self.get_config().agent_id,
        }

    def _understand_query_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """Use LLM to understand user intent."""
        if not self.llm:
            self.log("No LLM available, using fallback processing")
            return None

        try:
            prompt = get_query_understanding_prompt(query)
            response = self.llm.invoke([HumanMessage(content=prompt)])

            content = response.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()

            intent = json.loads(content)
            return intent

        except Exception as e:
            self.log_error(f"LLM query understanding failed: {e}")
            return None

    def _generate_rich_summary(self, movie_info: MovieInfo) -> str:
        """Generate LLM-enhanced movie summary for better searchability."""
        if not self.llm:
            return self._create_basic_summary(movie_info)

        try:
            movie_data = {
                "title": movie_info.title,
                "year": movie_info.year,
                "genre": movie_info.genre,
                "director": movie_info.director,
                "actors": movie_info.actors,
                "plot": movie_info.plot,
                "runtime": movie_info.runtime,
                "rating": movie_info.imdb_rating,
            }

            prompt = get_movie_summary_prompt(movie_data)
            response = self.llm.invoke([HumanMessage(content=prompt)])

            enhanced_summary = response.content.strip()
            self.log(f"Generated enhanced summary for {movie_info.title}")
            return enhanced_summary

        except Exception as e:
            self.log_error(f"Enhanced summary generation failed for {movie_info.title}: {e}")
            return self._create_basic_summary(movie_info)

    def _generate_storage_confirmation(self, successful: List, failed: List) -> Optional[str]:
        """Generate conversational confirmation message."""
        if not self.llm:
            return None

        try:
            prompt = get_storage_confirmation_prompt(successful, failed)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        except Exception as e:
            self.log_error(f"Confirmation generation failed: {e}")
            return None

    def _fallback_processing(self, query: str) -> dict:
        """Fallback processing when LLM is unavailable."""
        query_lower = query.lower()

        if any(pattern in query_lower for pattern in ["do i have", "is ", " in ", "check"]):
            title = query
            for pattern in ["do i have ", "is ", " in my watchlist", "check for "]:
                title = title.replace(pattern, "").strip()
            title = title.rstrip("?").strip()
            return self.check_movie_in_watchlist(title)

        elif any(pattern in query_lower for pattern in ["remove", "delete", "drop"]):
            title = query
            for pattern in ["remove ", "delete ", "drop "]:
                title = title.replace(pattern, "").strip()
            return self.delete_movie_from_watchlist(title)

        else:
            return {
                "message": "I can help you manage your movie watchlist. Try: 'Do I have Inception?' or 'Remove The Matrix'",
                "agent": self.get_config().agent_id,
            }

    def _convert_to_movie_info(self, movie) -> MovieInfo:
        """Convert movie data to MovieInfo object."""
        if isinstance(movie, MovieInfo):
            return movie
        return self._dict_to_movie_info(movie)

    def _dict_to_movie_info(self, movie_dict: dict) -> MovieInfo:
        """Convert TMDB API dict to MovieInfo object."""
        return MovieInfo(
            imdb_id=movie_dict.get("imdbID", ""),
            title=movie_dict.get("Title", ""),
            year=movie_dict.get("Year", ""),
            genre=movie_dict.get("Genre", ""),
            director=movie_dict.get("Director", ""),
            actors=movie_dict.get("Actors", ""),
            plot=movie_dict.get("Plot", ""),
            runtime=movie_dict.get("Runtime", ""),
            imdb_rating=movie_dict.get("imdbRating", "N/A"),
            poster_url=movie_dict.get("Poster", ""),
        )

    def _create_basic_summary(self, movie: MovieInfo) -> str:
        """Create a basic summary when LLM is unavailable."""
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
                "rating": movie.imdb_rating,
            },
        )

    def _get_movie_title(self, movie) -> str:
        """Get movie title for error reporting."""
        if isinstance(movie, MovieInfo):
            return f"{movie.title} ({movie.year})"
        else:
            return movie.get("Title", "Unknown") if hasattr(movie, "get") else "Unknown"

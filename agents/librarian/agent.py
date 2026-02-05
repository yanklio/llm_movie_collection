from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agents.base_agent import AgentConfig, BaseAgent
from agents.librarian.prompts import LIBRARIAN_SYSTEM_PROMPT
from storage.vector_store import VectorStore
from tools.movies.movie_info import MovieInfo


class Librarian(BaseAgent):
    """
    Watchlist Agent - Uses LLM with tools to manage and query the movie watchlist.
    
    The agent can recursively call tools to answer complex questions about
    the user's watchlist.
    """

    def __init__(
        self,
        model: str | None = None,
        verbose: bool = False,
        vector_store: VectorStore | None = None,
    ):
        super().__init__(model, verbose)
        self.vector_store = vector_store or VectorStore()
        self._setup_tools()

    def _setup_tools(self):
        """Create tools and bind them to LLM."""
        # Create tool functions that close over self.vector_store
        vs = self.vector_store
        
        @tool
        def search_watchlist(query: str) -> str:
            """Search the watchlist for movies matching a query (title, genre, director, actor, theme)."""
            results = vs.search(query, top_k=10)
            if not results:
                return "No movies found matching the search."
            
            movies = []
            for r in results:
                m = r.get("metadata", {})
                similarity = 1 - r.get("distance", 1)
                if similarity > 0.3:
                    movies.append(f"- {m.get('title', 'Unknown')} ({m.get('year', 'N/A')}) - {m.get('genre', 'N/A')}, directed by {m.get('director', 'Unknown')}")
            
            if not movies:
                return "No closely matching movies found."
            return f"Found {len(movies)} movies:\n" + "\n".join(movies)

        @tool
        def get_all_movies() -> str:
            """Get all movies in the watchlist."""
            all_movies = vs.get_all_movies()
            if not all_movies:
                return "The watchlist is empty."
            
            movies = []
            for r in all_movies:
                m = r.get("metadata", {})
                movies.append(f"- {m.get('title', 'Unknown')} ({m.get('year', 'N/A')}) - {m.get('genre', 'N/A')}")
            
            return f"Watchlist ({len(movies)} movies):\n" + "\n".join(movies)

        @tool  
        def count_movies() -> str:
            """Get the total number of movies in the watchlist."""
            count = vs.count()
            return f"There are {count} movies in the watchlist."

        @tool
        def delete_movie(title: str) -> str:
            """Delete a movie from the watchlist by title."""
            results = vs.search(title, top_k=1)
            if not results:
                return f"Movie '{title}' not found in watchlist."
            
            movie = results[0]
            imdb_id = movie.get("id")
            movie_title = movie.get("metadata", {}).get("title", title)
            
            if vs.delete_movie(imdb_id):
                return f"Successfully deleted '{movie_title}' from watchlist."
            return f"Failed to delete '{movie_title}'."

        self.tools = [search_watchlist, get_all_movies, count_movies, delete_movie]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    @classmethod
    def get_config(cls) -> AgentConfig:
        return AgentConfig(
            agent_id="movie_librarian",
            name="Movie Librarian",
            description="Watchlist management agent. Handles questions about your watchlist: checking movies, searching, counting, filtering, and deleting.",
            patterns=["check", "do i have", "watchlist", "delete", "remove"],
            capabilities=[
                "Check if movies exist in watchlist",
                "Search movies by any criteria",
                "Count and filter movies",
                "Delete movies from watchlist",
                "Answer complex watchlist questions",
            ],
            example_queries=[
                "Do I have Inception?",
                "How many Tom Hanks movies do I have?",
                "Remove The Matrix",
                "Show me my sci-fi movies",
            ],
        )

    def process(self, query: str) -> dict:
        """Process user query using tool-calling agent loop."""
        self.log(f"Processing: {query}")
        
        messages = [
            SystemMessage(content=LIBRARIAN_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        
        # Agent loop - let LLM call tools until it has an answer
        max_iterations = 5
        for i in range(max_iterations):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check if LLM wants to call tools
            if not response.tool_calls:
                self.log_success(f"Completed in {i+1} iteration(s)")
                return {
                    "query": query,
                    "response": response.content,
                    "operation_performed": "query",
                    "success": True,
                    "agent_id": self.get_config().agent_id,
                }
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                self.log(f"Tool: {tool_name}({tool_args})")
                
                # Execute the tool
                tool_fn = self.tool_map.get(tool_name)
                if tool_fn:
                    result = tool_fn.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"
                
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        
        return {
            "query": query,
            "response": "I couldn't complete the request. Please try again.",
            "operation_performed": "error",
            "success": False,
            "agent_id": self.get_config().agent_id,
        }

    def store_movies(self, movies: List[MovieInfo]) -> dict:
        """Store movies to watchlist - called by other agents."""
        self.log(f"Storing {len(movies)} movies")
        
        stored = []
        skipped = []
        
        for movie in movies:
            summary = f"Title: {movie.title} ({movie.year})\nGenre: {movie.genre}\nDirector: {movie.director}\nCast: {movie.actors}\nRating: {movie.imdb_rating}/10\nPlot: {movie.plot}"
            
            metadata = {
                "title": movie.title,
                "year": movie.year,
                "genre": movie.genre,
                "director": movie.director,
                "actors": movie.actors,
                "rating": movie.imdb_rating,
                "runtime": movie.runtime,
                "plot": movie.plot,
                "imdb_id": movie.imdb_id,
            }
            
            if self.vector_store.add_movie(movie.imdb_id, summary, metadata):
                stored.append(movie)
                self.log_success(f"Added: {movie.title}")
            else:
                skipped.append(movie)
                self.log(f"Skipped (exists): {movie.title}")
        
        return {
            "success": True,
            "stored_count": len(stored),
            "skipped_count": len(skipped),
            "successful": stored,
            "failed": skipped,
            "message": f"Stored {len(stored)} movies, skipped {len(skipped)} duplicates",
        }

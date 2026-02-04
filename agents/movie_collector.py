"""
MovieCollector Agent - TMDB API Integration Specialist

Responsibilities:
- Direct interaction with TMDB API
- Exposes movie search as MCP-style tools
- Provides movie data to other agents (especially Librarian)
"""

from typing import Optional
from langchain_core.tools import tool

from agents.base_agent import BaseAgent, AgentConfig
from tools.movies.movie_api import MovieAPITool
from tools.movies.movie_info import MovieInfo


class MovieCollector(BaseAgent):
    """
    API Integration Agent - Provides movie search capabilities.
    
    This agent is a pure API wrapper that other agents can use
    via MCP tools to fetch movie data.
    """
    
    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return agent configuration."""
        return AgentConfig(
            agent_id="movie_collector",
            name="Movie Collector",
            description="TMDB API integration specialist. Searches and retrieves movie data.",
            patterns=["fetch:", "lookup:", "search:"],
            keywords=["fetch", "lookup", "search", "get"],
            capabilities=[
                "Search movies by title",
                "Search movies by keyword",
                "Search movies by person (actor/director)",
                "Fetch detailed movie information"
            ],
            example_queries=[
                "Fetch Inception details",
                "Search for Tom Hanks movies",
                "Lookup Christopher Nolan films"
            ],
            requires_llm=False  # Pure API wrapper, no LLM needed
        )
    
    def __init__(
        self,
        model: str | None = None,
        movie_api: MovieAPITool | None = None,
        verbose: bool = False,
    ):
        """Initialize the MovieCollector."""
        super().__init__(model, verbose)
        self.movie_api = movie_api or MovieAPITool()
    
    def process(self, query: str) -> dict:
        """
        Process a search query.
        
        Args:
            query: Search query (title, person, keyword)
            
        Returns:
            Dictionary with search results
        """
        # This is a simple wrapper - actual searching happens via tools
        return {
            "agent": self.get_config().agent_id,
            "query": query,
            "message": "Use MCP tools: search_by_title, search_by_person, or search_by_keyword"
        }
    
    def create_mcp_tools(self):
        """
        Create MCP-style tools for other agents to use.
        
        Returns:
            List of tool functions
        """
        movie_api = self.movie_api
        
        @tool
        def search_by_title(title: str, year: Optional[str] = None) -> str:
            """
            Search for a specific movie by title.
            
            Args:
                title: Movie title (e.g., "Inception")
                year: Optional release year (e.g., "2010")
                
            Returns:
                JSON string with movie details including IMDb ID
            """
            try:
                result = movie_api.search_by_title(title, year)
                
                if not result:
                    return f"No movie found for title: {title}"
                
                return f"""Found: {result.get('Title')} ({result.get('Year')})
IMDb ID: {result.get('imdbID')}
Genre: {result.get('Genre', 'N/A')}
Plot: {result.get('Plot', 'N/A')}
Director: {result.get('Director', 'N/A')}
Cast: {result.get('Actors', 'N/A')}"""
            except Exception as e:
                return f"Error searching for '{title}': {e}"
        
        @tool
        def search_by_person(person_name: str) -> str:
            """
            Search for movies by actor or director name.
            
            Args:
                person_name: Full name (e.g., "Tom Hanks", "Christopher Nolan")
                
            Returns:
                List of movies with IMDb IDs and roles
            """
            try:
                results = movie_api.search_person(person_name)
                
                if not results:
                    return f"No movies found for: {person_name}"
                
                output = f"Found {len(results)} movie(s) for {person_name}:\n\n"
                for movie in results:
                    role = movie.get('Role', 'Unknown')
                    output += f"- {movie.get('imdbID')}: {movie.get('Title')} ({movie.get('Year')}) [{role}]\n"
                
                return output
            except Exception as e:
                return f"Error searching for '{person_name}': {e}"
        
        @tool
        def search_by_keyword(keyword: str, year: Optional[str] = None) -> str:
            """
            Search for multiple movies matching a keyword.
            
            Args:
                keyword: Search keyword (e.g., "Matrix", "sci-fi")
                year: Optional year filter
                
            Returns:
                List of matching movies with IMDb IDs
            """
            try:
                results = movie_api.search_movies(keyword, year)
                
                if not results:
                    return f"No movies found for keyword: {keyword}"
                
                output = f"Found {len(results)} movie(s):\n\n"
                for movie in results:
                    output += f"- {movie.get('imdbID')}: {movie.get('Title')} ({movie.get('Year')})\n"
                
                return output
            except Exception as e:
                return f"Error searching for '{keyword}': {e}"
        
        return [search_by_title, search_by_person, search_by_keyword]
    
    def search_by_title(self, title: str, year: Optional[str] = None) -> Optional[dict]:
        """Direct method for programmatic access."""
        return self.movie_api.search_by_title(title, year)
    
    def search_by_person(self, person_name: str) -> list[dict]:
        """Direct method for programmatic access."""
        return self.movie_api.search_person(person_name)
    
    def search_by_keyword(self, keyword: str, year: Optional[str] = None) -> list[dict]:
        """Direct method for programmatic access."""
        return self.movie_api.search_movies(keyword, year)

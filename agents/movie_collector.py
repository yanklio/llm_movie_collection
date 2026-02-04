from agents.base_agent import BaseAgent, AgentConfig
from tools.movies.movie_api import MovieAPITool


class MovieCollector(BaseAgent):
    """
    Data Collection Agent - Fetches movie data from TMDB API.
    
    Receives natural language queries and intelligently determines
    the best search strategy (title, person, keyword).
    """
    
    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return agent configuration."""
        return AgentConfig(
            agent_id="movie_collector",
            name="Movie Collector",
            description="Data fetching specialist. Searches TMDB API for movie information.",
            patterns=["fetch:", "fetch ", "lookup:", "lookup ", "get:", "get "],
            keywords=["fetch", "lookup", "get", "retrieve", "find"],
            capabilities=[
                "Search movies by title",
                "Search by person (actor/director)",
                "Search by keywords",
                "Retrieve detailed movie metadata"
            ],
            example_queries=[
                "Fetch Inception",
                "Get Brad Pitt movies",
                "Lookup sci-fi from 1999"
            ],
            requires_llm=False
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
        Process a natural language query to fetch movie data.
        
        Args:
            query: User's search query
            
        Returns:
            Dictionary with 'movies' list and metadata
        """
        self.log(f"Searching TMDB for: {query}")
        
        movies = []
        search_method = "unknown"
        
        try:
            # Strategy 1: Check for person-related keywords
            person_keywords = ["movies", "films", "filmography", "starring", "directed by", "actor", "director"]
            query_lower = query.lower()
            
            if any(kw in query_lower for kw in person_keywords):
                # Extract person name
                person_name = query_lower
                for kw in person_keywords:
                    person_name = person_name.replace(kw, "").strip()
                
                if person_name:
                    self.log(f"Person search: {person_name}")
                    results = self.movie_api.search_by_person(person_name)
                    if results:
                        movies = results
                        search_method = "person"
            
            # Strategy 2: Try title search if no results yet
            if not movies and len(query.split()) <= 5:  # Likely a title
                self.log(f"Title search: {query}")
                result = self.movie_api.search_by_title(query)
                if result:
                    movies = [result]
                    search_method = "title"
            
            # Strategy 3: Keyword search as fallback
            if not movies:
                self.log(f"Keyword search: {query}")
                results = self.movie_api.search_by_keyword(query)
                if results:
                    movies = results[:10]  # Limit to top 10
                    search_method = "keyword"
            
            if movies:
                self.log_success(f"✓ Found {len(movies)} movie(s) via {search_method} search")
            else:
                self.log("No movies found")
            
            return {
                "movies": movies,
                "count": len(movies),
                "query": query,
                "search_method": search_method,
                "agent": self.get_config().agent_id
            }
            
        except Exception as e:
            self.log_error(f"Search failed: {e}")
            return {
                "movies": [],
                "count": 0,
                "query": query,
                "error": str(e),
                "agent": self.get_config().agent_id
            }
    
    # Direct access methods for advanced use cases
    def search_by_title(self, title: str, year: str | None = None):
        """Direct title search."""
        return self.movie_api.search_by_title(title, year)
    
    def search_by_person(self, person: str):
        """Direct person search."""
        return self.movie_api.search_by_person(person)
    
    def search_by_keyword(self, keyword: str):
        """Direct keyword search."""
        return self.movie_api.search_by_keyword(keyword)

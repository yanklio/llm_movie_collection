import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.base_agent import AgentConfig, BaseAgent
from agents.movie_collector.prompts import (
    get_query_analysis_prompt,
    get_result_summary_prompt,
)
from tools.movies.movie_api import MovieAPITool


class MovieCollector(BaseAgent):
    """
    Data Collection Agent - Fetches movie data from TMDB API using LLM-enhanced query analysis.

    Uses LLM to intelligently analyze queries and determine the best search strategy,
    then provides conversational summaries of results.
    """

    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return agent configuration."""
        return AgentConfig(
            agent_id="movie_collector",
            name="Movie Collector",
            description="LLM-enhanced data fetching specialist. Analyzes queries intelligently and searches TMDB API.",
            patterns=["fetch:", "fetch ", "lookup:", "lookup ", "get:", "get "],
            capabilities=[
                "LLM-powered query analysis",
                "Smart search strategy selection",
                "Conversational result summaries",
                "Search by title, person, or keywords",
                "Detailed movie metadata retrieval",
            ],
            example_queries=[
                "Fetch Inception",
                "Get Brad Pitt movies from the 90s",
                "Lookup dark sci-fi thrillers",
                "Find Christopher Nolan films",
            ],
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

    def get_tools(self) -> List:
        """Return list of available tools for this agent."""
        return [
            self.search_movie_by_title,
            self.search_movies_by_person,
            self.search_movies_by_keyword,
            self.analyze_movie_query,
        ]

    @tool
    def search_movie_by_title(self, title: str, year: Optional[str] = None) -> Dict:
        """Search for a specific movie by title and optionally by year."""
        self.log(f"Direct title search: {title}")
        result = self.movie_api.search_by_title(title, year)
        return {"movie": result, "search_type": "title"}

    @tool
    def search_movies_by_person(self, person: str, limit: int = 10) -> Dict:
        """Search for movies by actor or director name."""
        self.log(f"Direct person search: {person}")
        results = self.movie_api.search_by_person(person)
        limited_results = results[:limit] if results else []
        return {"movies": limited_results, "search_type": "person", "count": len(limited_results)}

    @tool
    def search_movies_by_keyword(self, keyword: str, limit: int = 10) -> Dict:
        """Search for movies by keyword, genre, or theme."""
        self.log(f"Direct keyword search: {keyword}")
        results = self.movie_api.search_by_keyword(keyword)
        limited_results = results[:limit] if results else []
        return {"movies": limited_results, "search_type": "keyword", "count": len(limited_results)}

    @tool
    def analyze_movie_query(self, query: str) -> Dict:
        """Analyze a movie search query using LLM to determine the best search strategy."""
        analysis = self._analyze_query_with_llm(query)
        return analysis or {"error": "LLM analysis not available"}

    def process(self, query: str) -> dict:
        """Process a natural language query to fetch movie data with LLM enhancement."""
        self.log(f"Analyzing query with LLM: {query}")

        try:
            # Step 1: LLM-powered query analysis
            analysis = self._analyze_query_with_llm(query)
            if not analysis:
                return self._fallback_search(query)

            search_type = analysis.get("search_type", "keyword")
            search_terms = analysis.get("search_terms", query)
            confidence = analysis.get("confidence", 0.5)

            self.log(
                f"LLM Analysis: {search_type} search for '{search_terms}' (confidence: {confidence:.2f})"
            )

            # Step 2: Execute search based on LLM analysis
            movies = self._execute_search(search_type, search_terms)

            # Step 3: Generate conversational summary
            summary = self._generate_summary(query, search_type, movies) if movies else None

            result = {
                "movies": movies,
                "count": len(movies),
                "query": query,
                "search_method": search_type,
                "search_terms": search_terms,
                "confidence": confidence,
                "summary": summary,
                "agent": self.get_config().agent_id,
            }

            if movies:
                self.log_success(f"✓ Found {len(movies)} movie(s) via {search_type} search")
            else:
                self.log("No movies found")

            return result

        except Exception as e:
            self.log_error(f"Search failed: {e}")
            return {
                "movies": [],
                "count": 0,
                "query": query,
                "error": str(e),
                "agent": self.get_config().agent_id,
            }

    def _analyze_query_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze query and determine search strategy."""
        if not self.llm:
            self.log("No LLM available, using fallback analysis")
            return None

        try:
            prompt = get_query_analysis_prompt(query)
            response = self.llm.invoke([HumanMessage(content=prompt)])

            content = response.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()

            analysis = json.loads(content)
            return analysis

        except Exception as e:
            self.log_error(f"LLM query analysis failed: {e}")
            return None

    def _execute_search(self, search_type: str, search_terms: str) -> List[Dict]:
        """Execute the determined search strategy."""
        movies = []

        if search_type == "person":
            self.log(f"Person search: {search_terms}")
            results = self.movie_api.search_by_person(search_terms)
            if results:
                movies = results[:10]

        elif search_type == "title":
            self.log(f"Title search: {search_terms}")
            result = self.movie_api.search_by_title(search_terms)
            if result:
                movies = [result]

        else:  # keyword search
            self.log(f"Keyword search: {search_terms}")
            results = self.movie_api.search_by_keyword(search_terms)
            if results:
                movies = results[:10]

        return movies

    def _generate_summary(
        self, query: str, search_method: str, movies: List[Dict]
    ) -> Optional[str]:
        """Generate conversational summary of search results."""
        if not self.llm or not movies:
            return None

        try:
            movie_data = self._format_movies_for_summary(movies)

            prompt = get_result_summary_prompt(
                query=query,
                search_method=search_method,
                result_count=len(movies),
                movie_data=movie_data,
            )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        except Exception as e:
            self.log_error(f"Summary generation failed: {e}")
            return None

    def _format_movies_for_summary(self, movies: List[Dict]) -> str:
        """Format movie data for summary prompt."""
        formatted_movies = []

        for movie in movies[:5]:
            title = movie.get("Title", "Unknown")
            year = movie.get("Year", "N/A")
            genre = movie.get("Genre", "N/A")
            director = movie.get("Director", "Unknown")
            rating = movie.get("imdbRating", "N/A")

            formatted_movies.append(
                f"• {title} ({year}) - {genre}\n  Director: {director}, Rating: {rating}/10"
            )

        return "\n\n".join(formatted_movies)

    def _fallback_search(self, query: str) -> dict:
        """Fallback search when LLM is unavailable."""
        self.log("Using fallback heuristic search")

        query_lower = query.lower()
        movies = []
        search_method = "heuristic"

        # Check for person keywords
        person_keywords = ["movies", "films", "actor", "director", "starring"]
        if any(kw in query_lower for kw in person_keywords):
            person_name = query
            for kw in person_keywords:
                person_name = person_name.replace(kw, "").strip()

            results = self.movie_api.search_by_person(person_name)
            if results:
                movies = results[:10]
                search_method = "person"

        # Try title search for short queries
        elif len(query.split()) <= 4:
            result = self.movie_api.search_by_title(query)
            if result:
                movies = [result]
                search_method = "title"

        # Keyword search fallback
        if not movies:
            results = self.movie_api.search_by_keyword(query)
            if results:
                movies = results[:10]
                search_method = "keyword"

        return {
            "movies": movies,
            "count": len(movies),
            "query": query,
            "search_method": search_method,
            "agent": self.get_config().agent_id,
        }

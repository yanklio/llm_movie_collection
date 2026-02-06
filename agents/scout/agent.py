import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.base_agent import AgentConfig, BaseAgent
from agents.scout.prompts import (
    get_query_analysis_prompt,
    get_result_summary_prompt,
)
from tools.movies.movie_api import MovieAPITool
from utils.parsing import extract_json_from_response


class Scout(BaseAgent):
    """
    Data Collection Agent - Fetches movie data from TMDB API using LLM-enhanced query analysis.

    Uses LLM to intelligently analyze queries and determine the best search strategy,
    then provides conversational summaries of results.
    """

    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return agent configuration."""
        return AgentConfig(
            agent_id="scout",
            name="Scout",
            description="FIRST STEP for adding data to storage(movies). USE FOR: 'Add <movie>' commands (must fetch before storing), fetching movie data from TMDB, looking up movies not in watchlist. After fetching, movies must be stored via librarian.",
            patterns=["fetch", "lookup", "get", "add"],
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
        return {"items": limited_results, "search_type": "person", "count": len(limited_results)}

    @tool
    def search_movies_by_keyword(self, keyword: str, limit: int = 10) -> Dict:
        """Search for movies by keyword, genre, or theme."""
        self.log(f"Direct keyword search: {keyword}")
        results = self.movie_api.search_by_keyword(keyword)
        limited_results = results[:limit] if results else []
        return {"items": limited_results, "search_type": "keyword", "count": len(limited_results)}

    @tool
    def analyze_movie_query(self, query: str) -> Dict:
        """Analyze a movie search query using LLM to determine the best search strategy."""
        analysis = self._analyze_query_with_llm(query)
        return analysis or {"error": "LLM analysis not available"}

    def process(self, query: str) -> dict:
        """Process a natural language query to fetch movie data."""
        self.log(f"Processing query: {query}")

        try:
            analysis = self._analyze_query_with_llm(query)
            
            if analysis:
                search_type = analysis.get("search_type", "keyword")
                search_terms = analysis.get("search_terms", query)
                confidence = analysis.get("confidence", 0.5)
                self.log(f"Strategy: {search_type} search for '{search_terms}' (conf: {confidence:.2f})")
            else:
                return self._fallback_search(query)

            movies = self._execute_search(search_type, search_terms)
            
            summary = self._generate_summary(query, search_type, movies) if movies else None
            
            return self._build_result_dict(
                query=query,
                movies=movies,
                search_type=search_type,
                search_terms=search_terms,
                confidence=confidence,
                summary=summary
            )

        except Exception as e:
            self.log_error(f"Scout process failed: {e}")
            return self._build_error_result(query, str(e))

    def _build_result_dict(self, query, movies, search_type, search_terms, confidence, summary) -> dict:
        """Construct the standard result dictionary."""
        if movies:
            self.log_success(f"Found {len(movies)} movie(s)")
        else:
            self.log("No movies found")
            
        return {
            "items": movies,
            "count": len(movies),
            "query": query,
            "search_method": search_type,
            "search_terms": search_terms,
            "confidence": confidence,
            "summary": summary,
            "agent": self.get_config().agent_id,
            "success": len(movies) > 0,
        }

    def _build_error_result(self, query: str, error_msg: str) -> dict:
        """Construct an error result dictionary."""
        return {
            "items": [],
            "count": 0,
            "query": query,
            "error": error_msg,
            "agent": self.get_config().agent_id,
            "success": False
        }

    def _analyze_query_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze query and determine search strategy."""
        if not self.llm:
            self.log("No LLM available, using fallback analysis")
            return None

        try:
            prompt = get_query_analysis_prompt(query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Handle potential list response from some LLMs
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(part["text"])
                    else:
                        texts.append(str(part))
                content = " ".join(texts)
            if not isinstance(content, str):
                content = str(content)

            analysis = extract_json_from_response(content.strip())
            return analysis

        except Exception as e:
            self.log_error(f"LLM query analysis failed: {e}")
            return None

    def _execute_search(self, search_type: str, search_terms: str) -> List[Dict]:
        """Execute the determined search strategy."""
        search_map = {
            "person": self._search_person,
            "title": self._search_title,
            "keyword": self._search_keyword
        }
        
        handler = search_map.get(search_type, self._search_keyword)
        return handler(search_terms)

    def _search_person(self, query: str) -> List[Dict]:
        self.log(f"Person search: {query}")
        results = self.movie_api.search_by_person(query)
        return results[:10] if results else []

    def _search_title(self, query: str) -> List[Dict]:
        self.log(f"Title search: {query}")
        try:
            result = self.movie_api.search_by_title(query)
            return [result] if result else []
        except Exception:
            return []

    def _search_keyword(self, query: str) -> List[Dict]:
        self.log(f"Keyword search: {query}")
        results = self.movie_api.search_by_keyword(query)
        return results[:10] if results else []

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
            return self._clean_llm_response(response.content)
        except Exception as e:
            self.log_error(f"Summary generation failed: {e}")
            return None

    def _format_movies_for_summary(self, movies: List[Dict]) -> str:
        """Format movie data for summary prompt using normalized data."""
        lines = []
        for movie in movies[:5]:
            data = self._normalize_movie_data(movie)
            lines.append(
                f"• {data['title']} ({data['year']}) - {data['genre']}\n"
                f"  Director: {data['director']}, Rating: {data['rating']}/10"
            )
        return "\n\n".join(lines)

    def _normalize_movie_data(self, movie: Any) -> Dict[str, Any]:
        """Normalize movie object/dict into a standard dictionary."""
        if hasattr(movie, "title"):  
            return {
                "title": getattr(movie, "title", "Unknown"),
                "year": getattr(movie, "year", "N/A"),
                "genre": getattr(movie, "genre", "N/A"),
                "director": getattr(movie, "director", "Unknown"),
                "rating": getattr(movie, "imdb_rating", "N/A"),
            }
        
        return {
            "title": movie.get("Title", movie.get("title", "Unknown")),
            "year": movie.get("Year", movie.get("year", "N/A")),
            "genre": movie.get("Genre", movie.get("genre", "N/A")),
            "director": movie.get("Director", movie.get("director", "Unknown")),
            "rating": movie.get("imdbRating", movie.get("imdb_rating", "N/A")),
        }

    def _fallback_search(self, query: str) -> dict:
        """Fallback search when LLM is unavailable using a waterfall approach."""
        self.log("Using fallback heuristic search")
        
        movies, method = self._try_heuristic_person_search(query)
        
        if not movies:
            movies, method = self._try_heuristic_title_search(query)
            
        if not movies:
            movies, method = self._try_heuristic_keyword_search(query)
            
        return {
            "items": movies,
            "count": len(movies),
            "query": query,
            "search_method": method,
            "agent": self.get_config().agent_id,
        }

    def _try_heuristic_person_search(self, query: str) -> tuple[List, str]:
        keywords = ["movies", "films", "actor", "director", "starring"]
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in keywords):
            name = query
            for kw in keywords:
                name = name.replace(kw, "").strip()
            
            results = self.movie_api.search_by_person(name)
            if results:
                return results[:10], "person"
        return [], ""

    def _try_heuristic_title_search(self, query: str) -> tuple[List, str]:
        # Simple heuristic: short queries might be titles
        if len(query.split()) <= 4:
            try:
                result = self.movie_api.search_by_title(query)
                if result:
                    return [result], "title"
            except Exception:
                pass
        return [], ""

    def _try_heuristic_keyword_search(self, query: str) -> tuple[List, str]:
        results = self.movie_api.search_by_keyword(query)
        if results:
            return results[:10], "keyword"
        return [], "keyword"

    def _clean_llm_response(self, content: Any) -> str:
        """Clean LLM response content."""
        if isinstance(content, list):
            texts = [
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in content
            ]
            content = " ".join(texts)
        return str(content).strip()

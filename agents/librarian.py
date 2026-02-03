"""
Librarian Agent - Ingestion specialist with MCP tool integration.

Responsibilities:
- Expose MovieAPITool as MCP-style tools
- LLM orchestrates movie fetching and summarization
- Store enriched documents in Vector Database
"""

from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent
from tools.movie_api import MovieAPITool
from tools.movie_info import MovieInfo
from storage.vector_store import VectorStore


LIBRARIAN_SYSTEM_PROMPT = """You are an autonomous librarian agent for a movie knowledge base.

You have access to tools to search and retrieve movie data. You can process natural language requests like:
- "Add Inception"
- "Add all Christopher Nolan movies"
- "Find and add sci-fi movies from 1999"
- "Add movies with Tom Hanks"

**Available Tools:**
1. `search_movie_by_title` - Get detailed info for ONE specific movie (use for single titles)
2. `search_movies` - Find MULTIPLE movies matching a keyword (use for broad searches)
3. `search_person` - Find movies by ACTOR or DIRECTOR name (use for "Tom Hanks movies", "Nolan films")

**Your workflow:**
1. **Understand the request** - What is the user asking for?
2. **Use the RIGHT tool:**
   - For specific movie titles → `search_movie_by_title("Inception")`
   - For actor/director names → `search_person("Tom Hanks")` or `search_person("Christopher Nolan")`
   - For general keyword searches → `search_movies("Matrix")`
3. **Return movie list** - After using tools, return movies in this format:

```
MOVIES_TO_ADD:
- {imdb_id}: {title} ({year})
- {imdb_id}: {title} ({year})
...
```

**Examples:**

Request: "Add Inception"
Your response:
```
MOVIES_TO_ADD:
- tt1375666: Inception (2010)
```

Request: "Add Tom Hanks movies"
Steps:
1. Call search_person("Tom Hanks") - DO NOT use search_movies for actors!
2. Get the list of movies with IMDb IDs
3. Return them

Request: "Add Christopher Nolan movies"
Steps:
1. Call search_person("Christopher Nolan")
2. Get the list
3. Return it

CRITICAL: Always return MOVIES_TO_ADD with specific IMDb IDs. The system will handle the actual ingestion."""


class Librarian(BaseAgent):
    """
    Ingestion Agent - Uses LLM with MCP tools to enrich and store movie data.
    
    Process:
    1. LLM uses `search_movie` tool to fetch from OMDb
    2. LLM generates retrieval-optimized summary
    3. Agent stores in VectorStore with enriched metadata
    """
    
    def __init__(
        self,
        model: str | None = None,
        movie_api: MovieAPITool | None = None,
        vector_store: VectorStore | None = None,
    ):
        """Initialize the Librarian."""
        super().__init__(model)
        self.movie_api = movie_api or MovieAPITool()
        self.vector_store = vector_store or VectorStore()
        
        # Create MCP-style tools
        self.tools = self._create_tools()
        
        # Create ReAct agent
        self.agent = create_react_agent(
            self.llm,
            self.tools,
        )
        self.max_iterations = 5  # Limited iterations for ingestion
    
    def requires_llm(self) -> bool:
        """Librarian needs LLM for orchestration and summarization."""
        return True
    
    def _create_tools(self) -> list:
        """Create MCP-style tools from MovieAPITool."""
        
        # Capture self for closures
        movie_api = self.movie_api
        
        @tool
        def search_movie_by_title(title: str, year: str = "") -> str:
            """
            Search for a SINGLE specific movie by exact title.
            
            Args:
                title: Exact movie title (e.g. "Inception", "The Matrix")
                year: Optional year to narrow results (e.g. "2010")
            
            Returns:
                Detailed movie information including IMDb ID, director, cast, plot, rating.
            """
            try:
                year_param = year if year else None
                movie = movie_api.search_by_title(title, year=year_param)
                return f"""imdbID: {movie.imdb_id}
Title: {movie.title}
Year: {movie.year}
Genre: {movie.genre}
Director: {movie.director}
Cast: {movie.actors}
Runtime: {movie.runtime}
IMDb Rating: {movie.imdb_rating}/10
Plot: {movie.plot}"""
            except Exception as e:
                return f"Error: {e}"
        
        @tool
        def search_movies(keyword: str, year: str = "") -> str:
            """
            Search for MULTIPLE movies matching a keyword or search term.
            Returns a list of movies (up to 10 results).
            
            Args:
                keyword: Search keyword (director name, actor, movie title fragment, etc.)
                year: Optional year filter
            
            Returns:
                List of movies with Title, Year, and IMDb ID
            """
            try:
                year_param = year if year else None
                results = movie_api.search_movies(keyword, year=year_param)
                
                if not results:
                    return "No movies found matching that search."
                
                output = f"Found {len(results)} movie(s):\n\n"
                for movie in results[:10]:  # Limit to 10 results
                    output += f"- {movie.get('imdbID')}: {movie.get('Title')} ({movie.get('Year')})\n"
                
                return output
            except Exception as e:
                return f"Error: {e}"
        
        @tool
        def search_person(person_name: str) -> str:
            """
            Search for movies by PERSON NAME (actor, director, etc.).
            Use THIS tool for requests like "Tom Hanks movies" or "Christopher Nolan films".
            
            Args:
                person_name: Full name of the person (e.g. "Tom Hanks", "Christopher Nolan")
            
            Returns:
                List of movies the person acted in or directed, with IMDb IDs
            """
            try:
                results = movie_api.search_person(person_name)
                
                if not results:
                    return f"No movies found for person: {person_name}"
                
                output = f"Found {len(results)} movie(s) for {person_name}:\n\n"
                for movie in results:
                    role = movie.get('Role', 'Unknown')
                    output += f"- {movie.get('imdbID')}: {movie.get('Title')} ({movie.get('Year')}) [{role}]\n"
                
                return output
            except Exception as e:
                return f"Error: {e}"
        
        return [search_movie_by_title, search_movies, search_person]
    
    
    def process_request(self, request: str) -> tuple[list[MovieInfo], list[str]]:
        """
        Process a natural language request autonomously.
        
        Examples:
            - "Add Inception"
            - "Add all Christopher Nolan movies"
            - "Find sci-fi movies from 1999 and add them"
        
        Args:
            request: Natural language request
            
        Returns:
            (successful_movies, failed_queries)
        """
        self.log(f"Processing request: {request}")
        
        # Use LLM with tools to understand request and find movies
        messages = [
            SystemMessage(content=LIBRARIAN_SYSTEM_PROMPT),
            HumanMessage(content=request),
        ]
        
        try:
            result = self.agent.invoke(
                {"messages": messages},
                {"recursion_limit": 10}  # Allow more iterations for complex requests
            )
            
            # Advanced logging: Show tool calls
            self.log("\n=== Agent Execution Trace ===")
            for i, msg in enumerate(result["messages"]):
                msg_type = type(msg).__name__
                
                if msg_type == "AIMessage":
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            self.log(f"🔧 Tool Call: {tc.get('name')}({tc.get('args')})")
                
                elif msg_type == "ToolMessage":
                    content = msg.content[:150] if hasattr(msg, 'content') else ""
                    self.log(f"📥 Tool Response: {content}...")
            
            # Extract the response
            final_message = result["messages"][-1].content
            self.log(f"\n=== Final Response ===\n{final_message}\n")
            
            # Parse the MOVIES_TO_ADD section
            movie_ids = self._parse_movie_list(final_message)
            
            if not movie_ids:
                self.log_error("No movies identified from request")
                return [], []
            
            self.log_info(f"Found {len(movie_ids)} movie(s) to add")
            
            # Ingest each movie
            successful = []
            failed = []
            
            for imdb_id in movie_ids:
                try:
                    # Fetch and ingest by IMDb ID
                    movie_info = self.movie_api.search_by_imdb_id(imdb_id)
                    self._ingest_movie(movie_info)
                    successful.append(movie_info)
                except Exception as e:
                    self.log_error(f"Failed to add {imdb_id}: {e}")
                    failed.append(imdb_id)
            
            return successful, failed
            
        except Exception as e:
            self.log_error(f"Request processing failed: {e}")
            return [], []
    
    def _parse_movie_list(self, text: str) -> list[str]:
        """Extract IMDb IDs from LLM response."""
        import re
        
        # Look for pattern: - tt1234567: Title (Year)
        pattern = r'-\s*(tt\d+):'
        matches = re.findall(pattern, text)
        return matches
    
    def ingest(self, query: str, force_update: bool = False) -> MovieInfo:
        """
        Ingest a movie into the knowledge base using LLM orchestration.
        
        Args:
            query: Movie title, IMDb ID, or URL
            force_update: Replace existing entry
            
        Returns:
            MovieInfo object of ingested movie
        """
        self.log(f"Processing: {query}")
        
        # First, fetch the raw movie data (outside LLM to get MovieInfo object)
        try:
            movie_info = self.movie_api.search(query)
        except Exception as e:
            self.log_error(f"Failed to fetch: {e}")
            raise
        
        self.log_info(f"Found: {movie_info.title} ({movie_info.year})")
        
        # Check if already exists
        existing = self.vector_store.get_movie(movie_info.imdb_id)
        if existing and not force_update:
            self.log_error(f"Already in database: {movie_info.title}")
            raise ValueError(f"Movie '{movie_info.title}' already exists. Use --force to update.")
        
        # Use LLM with tools to generate enriched summary
        self.log("Generating retrieval-optimized summary via LLM...")
        try:
            enriched_doc = self._llm_summarize(query)
        except Exception as e:
            self.log_error(f"LLM summarization failed: {e}")
            # Fallback to basic format
            enriched_doc = movie_info.to_document()
        
        # Store in vector database
        metadata = movie_info.to_dict()
        metadata["enriched"] = True
        
        if force_update and existing:
            self.vector_store.delete_movie(movie_info.imdb_id)
        
        self.vector_store.add_movie(
            imdb_id=movie_info.imdb_id,
            document=enriched_doc,
            metadata=metadata,
        )
        
        self.log_success(f"Ingested: {movie_info.title}")
        return movie_info
    
    def _llm_summarize(self, query: str) -> str:
        """
        Use LLM with MCP tools to fetch and summarize movie data.
        
        The LLM will:
        1. Call search_movie tool
        2. Analyze the data
        3. Generate retrieval-optimized summary
        """
        messages = [
            SystemMessage(content=LIBRARIAN_SYSTEM_PROMPT),
            HumanMessage(content=f"Fetch and create a retrieval-optimized summary for: {query}"),
        ]
        
        result = self.agent.invoke(
            {"messages": messages},
            {"recursion_limit": self.max_iterations}
        )
        
        # Extract final response
        final_message = result["messages"][-1].content
        return final_message.strip()
    
    def _ingest_movie(self, movie_info: MovieInfo, force_update: bool = False):
        """
        Internal method to ingest a MovieInfo object.
        
        Generates enriched summary and stores in vector database.
        """
        # Check if already exists
        existing = self.vector_store.get_movie(movie_info.imdb_id)
        if existing and not force_update:
            self.log(f"Skipping (already exists): {movie_info.title}")
            return
        
        # Use basic document format (enrichment can be added later)
        enriched_doc = movie_info.to_document()
        
        # Store
        metadata = movie_info.to_dict()
        metadata["enriched"] = False  # Not using LLM summarization for batch
        
        if force_update and existing:
            self.vector_store.delete_movie(movie_info.imdb_id)
        
        self.vector_store.add_movie(
            imdb_id=movie_info.imdb_id,
            document=enriched_doc,
            metadata=metadata,
        )
        
        self.log_success(f"Added: {movie_info.title} ({movie_info.year})")
    
    def ingest_batch(
        self,
        queries: list[str],
        force_update: bool = False,
    ) -> tuple[list[MovieInfo], list[str]]:
        """
        Ingest multiple movies.
        
        Returns:
            (successful_movies, failed_queries)
        """
        successful = []
        failed = []
        
        for query in queries:
            try:
                movie = self.ingest(query, force_update=force_update)
                successful.append(movie)
            except Exception as e:
                self.log_error(f"Failed: {query} - {e}")
                failed.append(query)
        
        self.log_info(f"Batch complete: {len(successful)} added, {len(failed)} failed")
        return successful, failed
    
    def remove(self, imdb_id: str) -> bool:
        """Remove a movie from the knowledge base."""
        removed = self.vector_store.delete_movie(imdb_id)
        if removed:
            self.log_success(f"Removed: {imdb_id}")
        else:
            self.log_error(f"Not found: {imdb_id}")
        return removed

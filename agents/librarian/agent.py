from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agents.base_agent import AgentConfig, BaseAgent
from agents.librarian.prompts import LIBRARIAN_SYSTEM_PROMPT
from storage.vector_store import VectorStore


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
                    movies.append(
                        f"- {m.get('title', 'Unknown')} ({m.get('year', 'N/A')}) - {m.get('genre', 'N/A')}, directed by {m.get('director', 'Unknown')}"
                    )

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
                movies.append(
                    f"- {m.get('title', 'Unknown')} ({m.get('year', 'N/A')}) - {m.get('genre', 'N/A')}"
                )

            return f"Watchlist ({len(movies)} movies):\n" + "\n".join(movies)

        @tool
        def count_movies() -> str:
            """Get the total number of movies in the watchlist."""
            count = vs.count()
            return f"There are {count} movies in the watchlist."

        @tool
        def delete_movie(title: str) -> str:
            """Delete a movie from the watchlist by title."""
            results = vs.search(title, top_k=5)
            if not results:
                return f"Movie '{title}' not found in watchlist."

            target_title_clean = title.lower().strip()

            for movie in results:
                movie_title = movie.get("metadata", {}).get("title", "")
                if movie_title.lower().strip() == target_title_clean:
                    imdb_id = movie.get("id")
                    if vs.delete_movie(imdb_id):
                        return f"Successfully deleted '{movie_title}' from watchlist."
                    return f"Failed to delete '{movie_title}'."

            for movie in results:
                movie_title = movie.get("metadata", {}).get("title", "")
                movie_title_clean = movie_title.lower().strip()

                distance = movie.get("distance", 1.0)

                if (
                    target_title_clean in movie_title_clean
                    or movie_title_clean in target_title_clean
                ) and distance < 0.5:
                    imdb_id = movie.get("id")
                    if vs.delete_movie(imdb_id):
                        return f"Successfully deleted '{movie_title}' from watchlist (matched '{title}')."
                    return f"Failed to delete '{movie_title}'."

            top_match = results[0]
            if top_match.get("distance", 1.0) < 0.2:
                movie_title = top_match.get("metadata", {}).get("title", "")
                imdb_id = top_match.get("id")
                if vs.delete_movie(imdb_id):
                    return f"Successfully deleted '{movie_title}' from watchlist."

            found_titles = [m.get("metadata", {}).get("title", "") for m in results[:3]]
            return f"Could not find exact match for '{title}'. Did you mean one of these? {', '.join(found_titles)}"

        self.tools = [search_watchlist, get_all_movies, count_movies, delete_movie]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    @classmethod
    def get_config(cls) -> AgentConfig:
        return AgentConfig(
            agent_id="librarian",
            name="MLibrarian",
            description="STORAGE agent. USE FOR: checking if some entity exist ('do i have X?'), searching your existing content, counting entities, deleting entities.",
            patterns=["check", "do i have", "in my", "delete", "remove", "how many"],
            capabilities=[
                "Check if entities exist in storage",
                "Search existing storage",
                "Count and filter entities",
                "Delete entities from storage",
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

        max_iterations = 5
        executed_tools = set()
        deleted_count = 0

        for i in range(max_iterations):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                self.log_success(f"Completed in {i + 1} iteration(s)")
                return {
                    "query": query,
                    "response": response.content,
                    "operation_performed": "query",
                    "success": True,
                    "agent_id": self.get_config().agent_id,
                    "deleted_count": deleted_count,
                }

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                call_signature = f"{tool_name}:{str(tool_args)}"
                if call_signature in executed_tools:
                    self.log(f"Skipping duplicate call: {call_signature}")
                    result = f"Error: You already called {tool_name} with these arguments. Do not retry the same thing repeatedly."
                else:
                    self.log(f"Tool: {tool_name}({tool_args})")
                    executed_tools.add(call_signature)

                    tool_fn = self.tool_map.get(tool_name)
                    if tool_fn:
                        result = tool_fn.invoke(tool_args)
                        # Track deletions
                        if tool_name == "delete_movie" and "Successfully deleted" in str(result):
                            deleted_count += 1
                    else:
                        result = f"Unknown tool: {tool_name}"

                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        return {
            "query": query,
            "response": "I couldn't complete the request. Please try again.",
            "operation_performed": "error",
            "success": False,
            "agent_id": self.get_config().agent_id,
            "deleted_count": deleted_count,
        }

    def store_items(self, items: List[Any]) -> dict:
        """Store items (movies, books, etc) to watchlist - generic method."""
        self.log(f"Storing {len(items)} items")

        stored = []
        skipped = []

        import uuid

        for item in items:
            if hasattr(item, "to_dict"):
                data = item.to_dict()
                if hasattr(item, "to_document"):
                    document = item.to_document()
                else:
                    document = str(data)
            elif isinstance(item, dict):
                data = item
                document = str(data)
            else:
                self.log(f"Skipping unknown item type: {type(item)}")
                continue

            item_id = data.get("imdb_id") or data.get("id") or str(uuid.uuid4())
            title = data.get("title") or data.get("name") or "Unknown Title"

            if isinstance(item, dict):
                doc_parts = [f"Title: {title}"]
                for k, v in data.items():
                    if k not in ["title", "name", "id", "imdb_id", "poster_url"] and v:
                        doc_parts.append(f"{k.capitalize()}: {v}")
                document = "\n".join(doc_parts)

            safe_metadata = {}
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_metadata[k] = v
                else:
                    safe_metadata[k] = str(v)

            safe_metadata["title"] = title

            if self.vector_store.add_movie(item_id, document, safe_metadata):
                stored.append(title)
                self.log_success(f"Stored: {title}")
            else:
                skipped.append(title)
                self.log(f"Skipped (exists): {title}")

        return {
            "stored_count": len(stored),
            "skipped_count": len(skipped),
            "stored_titles": stored,
            "message": f"Successfully stored {len(stored)} items. Skipped {len(skipped)} duplicates.",
        }

    def store_movies(self, movies: List[Any]) -> dict:
        """Alias for store_items."""
        return self.store_items(movies)

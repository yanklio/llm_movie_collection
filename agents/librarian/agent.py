import uuid
from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agents.base_agent import AgentConfig, BaseAgent
from agents.librarian.prompts import LIBRARIAN_SYSTEM_PROMPT
from storage.vector_store import VectorStore


class Librarian(BaseAgent):
    """
    Librarian Agent - Uses LLM with tools to manage and query the user's entity collection.
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

    @tool
    def search_entities(self, query: str, entity_type: str = "movie") -> str:
        """
        Search for entities (movies, reviews) in the database.
        
        Args:
            query: The search query (e.g., "Inception", "funny sitcoms")
            entity_type: Type of entity to search for ("movie" or "review")
        """     
        results = self.vector_store.search(query, top_k=5, entity_type=entity_type)
        if not results:
            return "No entities found matching the search."
    
    def _find_entity(self, title: str) -> dict | None:
        """
        Find best matching entity by title.
        
        Returns entity dict with 'id', 'metadata', 'distance' or None.
        Uses progressive matching: exact → partial → semantic.
        """
        results = self.vector_store.search(title, top_k=5)
        if not results:
            return None

        target = title.lower().strip()

        for entity in results:
            entity_title = entity.get("metadata", {}).get("title", "").lower().strip()
            if entity_title == target:
                return entity

        for entity in results:
            entity_title = entity.get("metadata", {}).get("title", "").lower().strip()
            distance = entity.get("distance", 1.0)
            if (target in entity_title or entity_title in target) and distance < 0.5:
                return entity

        if results[0].get("distance", 1.0) < 0.2:
            return results[0]

        return None

    def _format_entity(self, entity: dict, include_creator: bool = True) -> str:
        """Format a single entity for display."""
        meta = entity.get("metadata", {})
        title = meta.get("title", "Unknown")
        year = meta.get("year", "N/A")
        genre = meta.get("genre", "N/A")
        
        base = f"{title} ({year}) - {genre}"
        if include_creator:
            creator = meta.get("director") or meta.get("author") or meta.get("creator", "Unknown")
            return f"{base}, by {creator}"
        return base

    def _get_similar_titles(self, title: str, max_results: int = 3) -> list[str]:
        """Get similar entity titles for suggestions."""
        results = self.vector_store.search(title, top_k=max_results)
        return [r.get("metadata", {}).get("title", "") for r in results if r]

    def _setup_tools(self):
        """Create tools and bind them to LLM."""
        vs = self.vector_store
        find_entity = self._find_entity
        format_entity = self._format_entity
        get_similar = self._get_similar_titles

        @tool
        def search_collection(query: str, limit: int = 10) -> str:
            """
            Search the collection for entities matching a query.
            
            Args:
                query: Search term - can be title, genre, creator, theme, or mood
                limit: Maximum number of results to return (default: 10)
            
            Returns matching entities with title, year, genre, and creator.
            """
            results = vs.search(query, top_k=limit)
            if not results:
                return "No entities found matching the search."

            matches = [r for r in results if r.get("distance", 2.0) < 1.45]
            
            if not matches:
                self.log(f"No matches found (closest dist: {results[0]['distance']:.3f} if results else 'None')")
                return "No closely matching entities found."

            formatted = "\n".join(f"- {format_entity(m)}" for m in matches)
            return f"Found {len(matches)} entities:\n{formatted}"

        @tool
        def get_all_entities(limit: int = 50) -> str:
            """
            Get all entities in the collection.
            
            Args:
                limit: Maximum number of entities to return (default: 50)
            
            Returns list of all entities with title, year, and genre.
            """
            all_entities = vs.get_all()
            if not all_entities:
                return "The collection is empty."

            entities = all_entities[:limit]
            formatted = "\n".join(
                f"- {format_entity(e, include_creator=False)}" for e in entities
            )
            
            total = len(all_entities)
            shown = len(entities)
            header = f"Collection ({total} entities)"
            if shown < total:
                header += f" - showing first {shown}"
            
            return f"{header}:\n{formatted}"

        @tool
        def count_entities() -> str:
            """Get the total number of entities in the collection."""
            count = vs.count()
            return f"There are {count} entities in the collection."

        @tool
        def delete_entity(title: str) -> str:
            """
            Delete an entity from the collection by title.
            
            Args:
                title: The title of the entity to delete (exact or partial match)
            
            Will try to find the best match and delete it.
            If no exact match, suggests similar titles.
            """
            entity = find_entity(title)
            
            if entity:
                entity_title = entity.get("metadata", {}).get("title", "Unknown")
                entity_id = entity.get("id")
                
                if vs.delete(entity_id):
                    return f"Successfully deleted '{entity_title}' from collection."
                return f"Failed to delete '{entity_title}' - please try again."

            suggestions = get_similar(title)
            if suggestions:
                return f"Entity '{title}' not found. Did you mean: {', '.join(suggestions)}?"
            return f"Entity '{title}' not found in collection."

        @tool
        def check_entity(title: str) -> str:
            """
            Check if a specific entity exists in the collection.
            
            Args:
                title: The title of the entity to check
            
            Returns whether the entity exists and its details if found.
            """
            entity = find_entity(title)
            
            if entity:
                return f"Yes, you have '{format_entity(entity)}' in your collection."
            
            suggestions = get_similar(title)
            if suggestions:
                return f"'{title}' is not in your collection. Similar items you have: {', '.join(suggestions)}"
            return f"'{title}' is not in your collection."

        @tool
        def add_review(movie_title: str, review_text: str) -> str:
            """
            Add a review for a movie.
            
            Args:
                movie_title: Title of the movie being reviewed
                review_text: The content of the review
            """
            movie = find_entity(movie_title)
            movie_meta = movie.get("metadata", {}) if movie else {}
            
            # Create review entity
            review_id = f"review_{uuid.uuid4()}"
            metadata = {
                "title": f"{movie_title} Review",
                "related_movie": movie_title,
                "entity_type": "review",
                "author": "User"
            }
            
            if movie_meta:
                if "genre" in movie_meta:
                    metadata["related_genre"] = movie_meta["genre"]
            
            if vs.add(review_id, review_text, metadata, entity_type="review"):
                return f"Successfully added review for '{movie_title}'."
            else:
                return "Failed to add review."

        self.tools = [search_collection, get_all_entities, count_entities, delete_entity, check_entity, add_review]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    @classmethod
    def get_config(cls) -> AgentConfig:
        return AgentConfig(
            agent_id="librarian",
            name="Librarian",
            description="STORAGE agent. USE FOR: checking if entities exist ('do I have X?'), searching your collection, counting entities, deleting entities, adding reviews.",
            patterns=["check", "do i have", "in my", "delete", "remove", "how many", "add review", "review of", "reviewed"],
            capabilities=[
                "Check if entities exist in storage",
                "Search existing collection",
                "Count and filter entities",
                "Delete entities from storage",
                "Add user reviews",
            ],
            example_queries=[
                "Do I have Inception?",
                "How many Tom Hanks movies do I have?",
                "Remove The Matrix",
                "Show me my sci-fi collection",
                "Add review for Inception: It was mind-bending!",
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
        executed_tools: set[str] = set()
        deleted_count = 0

        for iteration in range(max_iterations):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                self.log(f"Completed in {iteration + 1} iteration(s)", style="green")
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
                call_signature = f"{tool_name}:{tool_args}"

                if call_signature in executed_tools:
                    self.log(f"Skipping duplicate: {call_signature}")
                    result = f"Already called {tool_name} with these arguments. Try something different."
                    messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
                    continue

                executed_tools.add(call_signature)
                self.log(f"Tool: {tool_name}({tool_args})")

                tool_fn = self.tool_map.get(tool_name)
                if not tool_fn:
                    result = f"Unknown tool: {tool_name}"
                else:
                    result = tool_fn.invoke(tool_args)
                    if tool_name == "delete_entity" and "Successfully deleted" in str(result):
                        deleted_count += 1

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
        """Store items (entities) to collection."""
        self.log(f"Storing {len(items)} items")

        stored, skipped = [], []

        for item in items:
            if hasattr(item, "to_dict"):
                data = item.to_dict()
                document = item.to_document() if hasattr(item, "to_document") else str(data)
            elif isinstance(item, dict):
                data = {k.lower(): v for k, v in item.items()}
                document = self._build_document(data)
            else:
                self.log(f"Skipping unknown type: {type(item)}")
                continue

            item_id = data.get("imdb_id") or data.get("imdbid") or data.get("id") or str(uuid.uuid4())
            title = data.get("title") or data.get("name") or "Unknown Title"

            metadata = self._normalize_metadata(data)
            metadata["title"] = title
            
            entity_type = data.get("entity_type", "movie")

            if self.vector_store.add(item_id, document, metadata, entity_type=entity_type):
                stored.append(title)
                self.log_success(f"Stored: {title} ({entity_type})")
            else:
                skipped.append(title)
                self.log(f"Skipped (exists): {title}")

        return {
            "stored_count": len(stored),
            "skipped_count": len(skipped),
            "stored_titles": stored,
            "message": f"Successfully stored {len(stored)} items. Skipped {len(skipped)} duplicates.",
        }

    def _build_document(self, data: dict) -> str:
        """Build searchable document from entity data."""
        title = data.get("title") or data.get("name") or "Unknown"
        parts = [f"Title: {title}"]
        
        skip_keys = {"title", "name", "id", "imdb_id", "poster_url"}
        for key, value in data.items():
            if key not in skip_keys and value:
                parts.append(f"{key.capitalize()}: {value}")
        
        return "\n".join(parts)

    def _normalize_metadata(self, data: dict) -> dict:
        """Normalize metadata and cast numeric types."""
        normalized = {}
        for k, v in data.items():
            key = k.lower()
            if key == "year":
                try:
                    normalized[k] = int(str(v).split("-")[0]) # Handle "2023-05-12" or "2023"
                except:
                    normalized[k] = str(v)
            elif key in ["rating", "imdb_rating", "imdbRating"]:
                try:
                    normalized[k] = float(v)
                except:
                    normalized[k] = str(v)
            elif isinstance(v, (str, int, float, bool)):
                normalized[k] = v
            else:
                normalized[k] = str(v)
        return normalized

    def store_movies(self, movies: List[Any]) -> dict:
        """Alias for store_items (backward compatibility)."""
        return self.store_items(movies)

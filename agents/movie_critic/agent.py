import json
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import AgentConfig, BaseAgent
from agents.movie_critic.model import CriticRequest, CriticResponse, MovieResult
from agents.movie_critic.prompts import (
    CRITIC_SYSTEM_PROMPT,
    get_query_expansion_prompt,
    get_synthesis_prompt,
)
from storage.vector_store import VectorStore
from utils.parsing import extract_json_from_response


class MovieCritic(BaseAgent):
    """
    Retrieval Agent - Expands queries and synthesizes grounded recommendations.

    Process:
    1. Query Expansion: Convert vague moods → searchable concepts
    2. Similarity Search: Retrieve top-k relevant movies
    3. Synthesis: Generate conversational response grounded in retrieved context
    """

    @classmethod
    def get_config(cls) -> AgentConfig:
        return AgentConfig(
            agent_id="movie_critic",
            name="Movie Critic",
            description="RECOMMENDATION agent with DIRECT ACCESS to user's collection via RAG. USE FOR: recommendations ('recommend', 'suggest'), mood-based queries ('something dark', 'want something like'). Searches collection directly and provides COMPLETE answers - no need to call other agents after this one.",
            patterns=["recommend", "suggest", "want", "looking for", "find", "query"],
            capabilities=[
                "Query expansion for vague requests",
                "Semantic similarity search",
                "Personalized movie recommendations",
                "Mood-based suggestions",
            ],
            example_queries=[
                "Find dark sci-fi about dreams",
                "I want something emotional starring Tom Hanks",
                "Recommend action movies from the 90s",
                "Looking for mind-bending thrillers",
            ],
        )

    def __init__(
        self,
        model: str | None = None,
        vector_store: VectorStore | None = None,
        verbose: bool = False,
        top_k: int = 5,
    ):
        super().__init__(model, verbose)
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k

    def process(self, query: str) -> dict:
        request = CriticRequest(query=query, top_k=self.top_k)
        response = self.query_movies(request)
        return response.dict()

    def query_movies(self, request: CriticRequest) -> CriticResponse:
        self.log(f"Processing query: {request.query}")

        # 1. Expand Query
        expanded_query, filters = self._expand_query(request.query)
        self.log(f"Expanded query: {expanded_query}")
        
        # 2. Retrieve Movies
        movie_results = self._retrieve_movies(expanded_query, request.top_k, filters)
        movies = [MovieResult(**result) for result in movie_results]
        
        # 3. Retrieve Context (Reviews)
        review_results = self._retrieve_reviews(expanded_query)

        # 4. Generate Response
        if not movies:
            self.log_error("No movies found in knowledge base")
            response_text = "I don't have any movies in my database yet. Please add some movies first!"
        else:
            self.log_success(f"Found {len(movies)} relevant movies")
            response_text = self._synthesize(request.query, movie_results, review_results)

        return CriticResponse(
            query=request.query,
            expanded_query=expanded_query,
            movies=movies,
            response=response_text,
        )

    def _retrieve_movies(self, query: str, top_k: int, filters: dict) -> list[dict]:
        """Execute vector search for movies."""
        return self.vector_store.search(
            query, 
            top_k=top_k, 
            entity_type="movie",
            filters=filters
        )

    def _retrieve_reviews(self, query: str) -> list[dict]:
        """Execute vector search for reviews."""
        results = self.vector_store.search(query, top_k=3, entity_type="review")
        if results:
            self.log_success(f"Found {len(results)} relevant reviews for context")
        return results

    def _expand_query(self, user_query: str) -> tuple[str, dict]:
        prompt = get_query_expansion_prompt(user_query)

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            data = extract_json_from_response(content)
            
            search_terms = data.get("search_terms", user_query)
            filters = data.get("filters", {})
            
            final_filters = {k: v for k, v in filters.items() if v is not None}
            
            return search_terms, final_filters
            
        except Exception as e:
            self.log_error(f"Query expansion failed: {e}")
            self.log_error(f"Failed content: {content!r}")
            return user_query, {}

    def _synthesize(self, user_query: str, results: list[dict], reviews: list[dict] = None) -> str:
        if not self.llm:
            return self._fallback_response(results)

        context = self._build_context(results, reviews)
        content = get_synthesis_prompt(user_query, context)

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            self.log_error(f"Synthesis failed: {e}")
            return self._fallback_response(results)

    def _build_context(self, results: list[dict], reviews: list[dict] = None) -> str:
        context_parts = []
        
        if reviews:
            context_parts.append("RELEVANT USER REVIEWS:")
            for i, review in enumerate(reviews, 1):
                doc = review.get("document", "")
                meta = review.get("metadata", {})
                title = meta.get("title", "Unknown")
                context_parts.append(f"Review {i} ({title}):\n{doc}\n---")
            context_parts.append("\nAVAILABLE MOVIES:")

        for i, result in enumerate(results, 1):
            document = result.get("document", "")
            similarity_score = result.get("distance", 0)
            
            meta = result.get("metadata", {})
            title = meta.get("title", "Unknown")
            year = meta.get("year", "N/A")
            genre = meta.get("genre", "N/A")

            context_parts.append(
                f"Movie {i} - {title} ({year}) [{genre}]:\n{document}\n\nSimilarity Score: {similarity_score:.3f}\n---"
            )

        return "\n".join(context_parts)

    def _fallback_response(self, results: list[dict]) -> str:
        lines = ["Here are some movies that might interest you:\n"]

        for result in results:
            metadata = result.get("metadata", {})
            title = metadata.get("title", "Unknown")
            year = metadata.get("year", "N/A")
            genre = metadata.get("genre", "N/A")

            lines.append(f"• {title} ({year}) - {genre}")

        return "\n".join(lines)

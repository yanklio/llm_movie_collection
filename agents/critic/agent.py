from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import AgentConfig, BaseAgent
from agents.critic.model import CriticRequest, CriticResponse, MovieResult
from agents.critic.prompts import (
    CRITIC_SYSTEM_PROMPT,
    get_query_expansion_prompt,
    get_synthesis_prompt,
)
from storage.vector_store import VectorStore


class Critic(BaseAgent):
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
            description="Movie recommendation and search agent. USE FOR: recommendations ('recommend', 'suggest'), searching existing watchlist ('find', 'want', 'looking for'), mood-based queries ('something dark', 'feeling emotional'), preference queries ('similar to X', 'I need'). Searches your personal watchlist and provides personalized suggestions.",
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

        expanded_query = self._expand_query(request.query)
        self.log(f"Expanded query: {expanded_query}")

        results = self.vector_store.search(expanded_query, top_k=request.top_k)

        movies = [MovieResult(**result) for result in results]

        if not movies:
            self.log_error("No movies found in knowledge base")
            response_text = (
                "I don't have any movies in my database yet. Please add some movies first!"
            )
        else:
            self.log_success(f"Found {len(movies)} relevant movies")
            response_text = self._synthesize(request.query, results)

        return CriticResponse(
            query=request.query,
            expanded_query=expanded_query,
            movies=movies,
            response=response_text,
        )

    def _expand_query(self, user_query: str) -> str:
        prompt = get_query_expansion_prompt(user_query)

        try:
            response = self.llm.invoke(prompt)
            expanded = response.content.strip()
            return expanded if expanded else user_query
        except Exception as e:
            self.log_error(f"Query expansion failed: {e}")
            return user_query

    def _synthesize(self, user_query: str, results: list[dict]) -> str:
        if not self.llm:
            return self._fallback_response(results)

        context = self._build_context(results)
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

    def _build_context(self, results: list[dict]) -> str:
        context_parts = []

        for i, result in enumerate(results, 1):
            document = result.get("document", "")
            similarity_score = result.get("distance", 0)

            context_parts.append(
                f"Movie {i}:\n{document}\n\nSimilarity Score: {similarity_score:.3f}\n---"
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

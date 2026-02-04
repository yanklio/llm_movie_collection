from langchain_core.messages import SystemMessage, HumanMessage

from agents.base_agent import BaseAgent, AgentConfig
from storage.vector_store import VectorStore


CRITIC_SYSTEM_PROMPT = """You are a knowledgeable movie critic and recommendation assistant.

Your role is to help users find movies based on their preferences, moods, or vague descriptions.

**Your workflow:**
1. **Understand the request** - What kind of movies is the user looking for?
2. **Review the provided context** - You will be given relevant movies from the database
3. **Provide recommendations** - Based ONLY on the provided context, suggest movies that match the user's request

**CRITICAL RULES:**
- ONLY recommend movies that are provided in the context
- DO NOT make up or hallucinate movies
- If no good matches are found, be honest about it
- Focus on the mood, themes, and feel of the movies
- Be conversational and helpful

**Response format:**
For each recommended movie, provide:
- Title and year
- Why it matches the user's request (focus on mood, themes, style)
- Brief description highlighting relevant aspects

Keep responses natural and conversational."""


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
        """Return agent configuration for routing."""
        return AgentConfig(
            agent_id="movie_critic",
            name="Movie Critic",
            description="Intelligent movie recommendation agent. Provides personalized recommendations based on mood, genre, or themes.",
            patterns=["find:", "find ", "query:", "query ", "search:", "search ", "recommend:", "recommend "],
            keywords=["find", "query", "search", "recommend", "suggest", "want", "looking for"],
            capabilities=[
                "Query expansion for vague requests",
                "Semantic similarity search",
                "Personalized movie recommendations",
                "Mood-based suggestions"
            ],
            example_queries=[
                "Find dark sci-fi about dreams",
                "I want something emotional starring Tom Hanks",
                "Recommend action movies from the 90s",
                "Looking for mind-bending thrillers"
            ],
            requires_llm=True
        )
    
    def __init__(
        self,
        model: str | None = None,
        vector_store: VectorStore | None = None,
        verbose: bool = False,
        top_k: int = 5,
    ):
        """Initialize the Critic."""
        super().__init__(model, verbose)
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
    
    def process(self, query: str) -> dict:
        """
        Process a query (BaseAgent interface).
        
        Args:
            query: User's request
            
        Returns:
            Dictionary with 'response' and metadata
        """
        response = self.query(query)
        return {
            "response": response,
            "agent": self.get_config().agent_id
        }
    
    def query(self, user_query: str) -> str:
        """
        Process a user query and return movie recommendations.
        
        Args:
            user_query: Natural language query (e.g., "dark sci-fi about dreams")
            
        Returns:
            Conversational response with recommendations
        """
        self.log(f"Processing query: {user_query}")
        
        expanded_query = self._expand_query(user_query)
        self.log(f"Expanded query: {expanded_query}")
        
        results = self.vector_store.search(expanded_query, top_k=self.top_k)
        
        if not results:
            self.log_error("No movies found in knowledge base")
            return "I don't have any movies in my database yet. Please add some movies first!"
        
        self.log_success(f"Found {len(results)} relevant movies")
        
        response = self._synthesize(user_query, results)
        
        return response
    
    def _expand_query(self, user_query: str) -> str:
        """
        Expand vague user queries into more searchable concepts.
        
        Examples:
        - "I'm feeling sad" → "melancholic, emotional drama, loss, grief"
        - "something thrilling" → "suspense, tension, mystery, thriller"
        """
        if not self.llm:
            return user_query
        
        prompt = f"""Expand this user query into searchable movie concepts.

User query: "{user_query}"

Extract key concepts like:
- Genres (action, sci-fi, drama, etc.)
- Moods (dark, uplifting, tense, melancholic, etc.)
- Themes (love, loss, redemption, dreams, etc.)
- Styles (noir, dystopian, mind-bending, etc.)

Respond with a comma-separated list of searchable keywords.
Example: "dark, sci-fi, mind-bending, dreams, reality"

Expanded query:"""

        try:
            response = self.llm.invoke(prompt)
            expanded = response.content.strip()
            return expanded if expanded else user_query
        except Exception as e:
            self.log_error(f"Query expansion failed: {e}")
            return user_query
    
    def _synthesize(self, user_query: str, results: list[dict]) -> str:
        """
        Synthesize a conversational response grounded in retrieved movies.
        
        Args:
            user_query: Original user query
            results: Retrieved movies from VectorStore
            
        Returns:
            Conversational recommendation response
        """
        if not self.llm:
            return self._fallback_response(results)
        
        context = self._build_context(results)
        
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=f"""User Query: "{user_query}"

Retrieved Movies from Database:
{context}

Based on these movies, provide your recommendations."""),
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            self.log_error(f"Synthesis failed: {e}")
            return self._fallback_response(results)
    
    def _build_context(self, results: list[dict]) -> str:
        """Build context string from retrieved movies."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            document = result.get("document", "")
            
            context_parts.append(f"""Movie {i}:
{document}

Similarity Score: {result.get('distance', 0):.3f}
---""")
        
        return "\n".join(context_parts)
    
    def _fallback_response(self, results: list[dict]) -> str:
        """Simple fallback response when LLM is unavailable."""
        lines = ["Here are some movies that might interest you:\n"]
        
        for result in results:
            metadata = result.get("metadata", {})
            title = metadata.get("title", "Unknown")
            year = metadata.get("year", "N/A")
            genre = metadata.get("genre", "N/A")
            
            lines.append(f"• {title} ({year}) - {genre}")
        
        return "\n".join(lines)

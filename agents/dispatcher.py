"""
Dispatcher Agent - Intent classification and routing.

Responsibilities:
- Analyze user intent (add movies vs. query for recommendations)
- Hybrid routing: Deterministic patterns + LLM classification
- Route requests to Librarian or Critic
"""

from enum import Enum

from agents.base import BaseAgent


class Intent(Enum):
    """User intent types."""
    ADD_MOVIE = "add"        # Add movies to the knowledge base
    QUERY_MOVIES = "query"   # Query for movie recommendations
    UNKNOWN = "unknown"


class Dispatcher(BaseAgent):
    """
    Routing Agent - Classifies user intent and determines which agent to use.
    
    Uses hybrid approach:
    1. Deterministic: Direct command patterns (add:, find:, query:)
    2. Semantic: LLM classification for natural language
    """
    
    def __init__(self, model: str | None = None):
        """Initialize the Dispatcher."""
        super().__init__(model)
        
        # Deterministic patterns for bypassing LLM
        self.add_patterns = [
            "add:", "add ", "ingest:", "ingest ",
            "store:", "store ", "save:", "save "
        ]
        self.query_patterns = [
            "find:", "find ", "query:", "query ",
            "search:", "search ", "recommend:", "recommend ",
            "suggest:", "suggest ", "show:", "show "
        ]
    
    def requires_llm(self) -> bool:
        """Dispatcher can work without LLM using deterministic routing."""
        return False
    
    def route(self, user_input: str) -> Intent:
        """
        Classify user intent and determine routing.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Intent enum (ADD_MOVIE or QUERY_MOVIES)
        """
        user_lower = user_input.lower().strip()
        
        # 1. Deterministic routing (fast path)
        for pattern in self.add_patterns:
            if user_lower.startswith(pattern):
                self.log(f"📌 Deterministic routing → LIBRARIAN (pattern: '{pattern}')")
                return Intent.ADD_MOVIE
        
        for pattern in self.query_patterns:
            if user_lower.startswith(pattern):
                self.log(f"📌 Deterministic routing → CRITIC (pattern: '{pattern}')")
                return Intent.QUERY_MOVIES
        
        # 2. Heuristic routing (keyword-based)
        add_keywords = ["add", "ingest", "store", "save", "import"]
        query_keywords = ["find", "recommend", "suggest", "want", "looking for", "show me"]
        
        has_add_keyword = any(keyword in user_lower for keyword in add_keywords)
        has_query_keyword = any(keyword in user_lower for keyword in query_keywords)
        
        if has_add_keyword and not has_query_keyword:
            self.log("🔍 Heuristic routing → LIBRARIAN (add keyword detected)")
            return Intent.ADD_MOVIE
        
        if has_query_keyword and not has_add_keyword:
            self.log("🔍 Heuristic routing → CRITIC (query keyword detected)")
            return Intent.QUERY_MOVIES
        
        # 3. Semantic routing (LLM classification) - only if both or neither keywords
        if self.llm:
            self.log("🤖 Using LLM for intent classification...")
            return self._llm_classify(user_input)
        
        # 4. Default fallback: assume query
        self.log("⚠️  No clear intent, defaulting to CRITIC")
        return Intent.QUERY_MOVIES
    
    def _llm_classify(self, user_input: str) -> Intent:
        """
        Use LLM to classify ambiguous user intent.
        """
        prompt = f"""Classify the user's intent into one of these categories:

1. **ADD_MOVIE**: User wants to add movies to the database
   Examples: "Add Inception", "Store The Matrix", "I want to add Tom Hanks movies"
   
2. **QUERY_MOVIES**: User wants recommendations or to search existing movies  
   Examples: "Find me a dark sci-fi", "I want something like Inception", "Show me action movies"

User input: "{user_input}"

Respond with ONLY one word: ADD_MOVIE or QUERY_MOVIES"""

        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip().upper()
            
            if "ADD_MOVIE" in result or "ADD" in result:
                self.log_success("LLM classified as: ADD_MOVIE → LIBRARIAN")
                return Intent.ADD_MOVIE
            elif "QUERY_MOVIES" in result or "QUERY" in result:
                self.log_success("LLM classified as: QUERY_MOVIES → CRITIC")
                return Intent.QUERY_MOVIES
            else:
                self.log_error(f"Unexpected LLM response: {result}, defaulting to CRITIC")
                return Intent.QUERY_MOVIES
                
        except Exception as e:
            self.log_error(f"LLM classification failed: {e}, defaulting to CRITIC")
            return Intent.QUERY_MOVIES
    
    def extract_query(self, user_input: str) -> str:
        """
        Extract the actual query from user input, removing command prefixes.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Clean query string
        """
        user_lower = user_input.lower().strip()
        
        # Remove deterministic prefixes
        all_patterns = self.add_patterns + self.query_patterns
        for pattern in all_patterns:
            if user_lower.startswith(pattern):
                # Remove the pattern prefix
                query = user_input[len(pattern):].strip()
                return query
        
        # No prefix found, return as-is
        return user_input.strip()

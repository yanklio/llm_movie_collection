from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END

from agents.base_agent import BaseAgent, AgentConfig
from agents.registry import AgentRegistry


class OrchestrationState(TypedDict):
    """State for multi-agent orchestration."""
    query: str                      
    query_lower: str                
    
    intent: str                     
    target_agent_id: str            
    routing_method: str             
    confidence: float              
    
    movie_data: Optional[dict]      
    storage_result: Optional[dict]  
    
    result: Optional[dict]          
    clean_query: str                


class Dispatcher(BaseAgent):
    """
    Orchestration Agent - Coordinates multi-agent workflows.
    
    For "Add movie" requests:
    1. Route intent → add_movie
    2. Call MovieCollector → fetch movie data
    3. Call Librarian → store in vector DB
    4. Return results
    
    For "Find movie" requests:
    1. Route intent → query_movie
    2. Call Critic → get recommendations
    3. Return results
    """
    
    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return dispatcher configuration."""
        return AgentConfig(
            agent_id="dispatcher",
            name="Dispatcher",
            description="Multi-agent orchestrator. Coordinates workflows between MovieCollector, Librarian, and Critic.",
            patterns=[],
            keywords=[],
            capabilities=[
                "Intent classification",
                "Multi-agent orchestration",
                "Workflow coordination"
            ],
            example_queries=[],
            requires_llm=True  # Need LLM for intent classification
        )
    
    def __init__(self, model: str | None = None, verbose: bool = False):
        """Initialize the Dispatcher."""
        super().__init__(model, verbose)
        self.agent_configs = AgentRegistry.get_configs()
        self.graph = self._build_graph()
    
    def _classify_intent_with_llm(self, query: str) -> tuple[str, str, float]:
        """
        Use LLM to classify user intent.
        
        Returns:
            (intent, clean_query, confidence)
        """
        prompt = f"""Classify the user's intent for this movie-related query.

User query: "{query}"

Intents:
- add_movie: User wants to add/save/store a movie (e.g., "add Inception", "put Matrix to watchlist", "I want to watch Pulp Fiction")
- query_movie: User wants to find/search/get recommendations (e.g., "find dark sci-fi", "recommend action movies", "something emotional")
- fetch_movie: User wants to lookup/fetch movie information without adding (e.g., "fetch Inception details", "get info about The Matrix")

Respond ONLY with a JSON object in this exact format:
{{
  "intent": "add_movie" or "query_movie" or "fetch_movie",
  "clean_query": "the movie title or search terms without action words",
  "confidence": 0.0 to 1.0
}}

Examples:
- "Put Inception to watchlist" → {{"intent": "add_movie", "clean_query": "Inception", "confidence": 0.95}}
- "Find dark sci-fi" → {{"intent": "query_movie", "clean_query": "dark sci-fi", "confidence": 0.9}}
- "Fetch The Matrix" → {{"intent": "fetch_movie", "clean_query": "The Matrix", "confidence": 0.85}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Extract JSON from response
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
                intent = result.get("intent", "query_movie")
                clean_query = result.get("clean_query", query)
                confidence = float(result.get("confidence", 0.7))
                
                if self.verbose:
                    self.log(f"[VERBOSE] LLM classified: intent={intent}, confidence={confidence:.2f}")
                
                return intent, clean_query, confidence
            
        except Exception as e:
            self.log_error(f"LLM classification failed: {e}")
        
        # Fallback to pattern matching
        return "query_movie", query, 0.3
    
    def _build_graph(self) -> StateGraph:
        """Build the orchestration state graph."""
        workflow = StateGraph(OrchestrationState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("fetch_movie_data", self._fetch_movie_data)
        workflow.add_node("store_movie", self._store_movie)
        workflow.add_node("query_movies", self._query_movies)
        workflow.add_node("finalize", self._finalize)
        
        # Entry point
        workflow.set_entry_point("classify_intent")
        
        # Routing based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            lambda state: state["intent"],
            {
                "add_movie": "fetch_movie_data",
                "query_movie": "query_movies",
                "fetch_movie": "fetch_movie_data",
            }
        )
        
        # Add movie workflow: fetch → store → finalize
        workflow.add_edge("fetch_movie_data", "store_movie")
        workflow.add_edge("store_movie", "finalize")
        
        # Query workflow: query → finalize
        workflow.add_edge("query_movies", "finalize")
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def process(self, query: str) -> dict:
        """
        Orchestrate a multi-agent workflow.
        
        Args:
            query: User's request
            
        Returns:
            Dictionary with results
        """
        # Initialize state
        initial_state: OrchestrationState = {
            "query": query,
            "query_lower": query.lower().strip(),
            "intent": "unknown",
            "target_agent_id": "",
            "routing_method": "unknown",
            "confidence": 0.0,
            "movie_data": None,
            "storage_result": None,
            "result": None,
            "clean_query": query,
        }
        
        # Execute orchestration graph
        result_state = self.graph.invoke(initial_state)
        
        return result_state.get("result", {})
    
    def _classify_intent(self, state: OrchestrationState) -> OrchestrationState:
        """Classify user intent using patterns first, then LLM for complex queries."""
        query_lower = state["query_lower"]
        
        # Pattern matching for explicit intents
        add_patterns = ["add:", "add ", "ingest:", "ingest ", "store:", "store "]
        query_patterns = ["find:", "find ", "query:", "query ", "search:", "search "]
        fetch_patterns = ["fetch:", "fetch ", "lookup:", "lookup "]
        
        # Check explicit patterns first (high confidence)
        for pattern in add_patterns:
            if query_lower.startswith(pattern):
                state["intent"] = "add_movie"
                state["routing_method"] = "pattern"
                state["confidence"] = 1.0
                state["clean_query"] = state["query"][len(pattern):].strip()
                self.log(f"Intent: add_movie (pattern: '{pattern}')")
                return state
        
        for pattern in fetch_patterns:
            if query_lower.startswith(pattern):
                state["intent"] = "fetch_movie"
                state["routing_method"] = "pattern"
                state["confidence"] = 1.0
                state["clean_query"] = state["query"][len(pattern):].strip()
                self.log(f"Intent: fetch_movie (pattern: '{pattern}')")
                return state
        
        for pattern in query_patterns:
            if query_lower.startswith(pattern):
                state["intent"] = "query_movie"
                state["routing_method"] = "pattern"
                state["confidence"] = 1.0
                state["clean_query"] = state["query"][len(pattern):].strip()
                self.log(f"Intent: query_movie (pattern: '{pattern}')")
                return state
        
        # No explicit pattern - use LLM for natural language understanding
        self.log("No explicit pattern found, using LLM classification...")
        intent, clean_query, confidence = self._classify_intent_with_llm(state["query"])
        
        state["intent"] = intent
        state["routing_method"] = "llm"
        state["confidence"] = confidence
        state["clean_query"] = clean_query
        self.log(f"Intent: {intent} (LLM, confidence: {confidence:.2f})")
        
        return state
    
    def _fetch_movie_data(self, state: OrchestrationState) -> OrchestrationState:
        """Call MovieCollector to fetch movie data."""
        self.log_success("Step 1: Fetching movie data via MovieCollector...")
        
        # Get MovieCollector agent
        collector_class = AgentRegistry.get_agent("movie_collector")
        if not collector_class:
            state["movie_data"] = {"error": "MovieCollector not available", "movies": []}
            return state
        
        collector = collector_class(verbose=self.verbose)
        
        if self.verbose:
            self.log(f"[VERBOSE] Calling MovieCollector.process('{state['clean_query']}')")
        
        # MovieCollector agent processes the query intelligently
        result = collector.process(state['clean_query'])
        
        # Extract movies from result
        movies = result.get("movies", [])
        state["movie_data"] = {
            "movies": movies,
            "count": len(movies),
            "query": state['clean_query'],
            "search_method": result.get("search_method", "unknown")
        }
        
        if result.get("error"):
            self.log_error(f"✗ Fetch failed: {result['error']}")
        else:
            self.log_success(f"✓ Fetched {len(movies)} movie(s) via {result.get('search_method', 'unknown')}")
        
        return state
    
    def _store_movie(self, state: OrchestrationState) -> OrchestrationState:
        """Call Librarian to store movie data."""
        movie_data = state.get("movie_data", {})
        movies = movie_data.get("movies", [])
        
        if not movies:
            state["storage_result"] = {"error": "No movie data to store", "successful": [], "failed": []}
            return state
        
        self.log_success(f"Step 2: Storing {len(movies)} movie(s) via Librarian...")
        
        # Get Librarian agent
        librarian_class = AgentRegistry.get_agent("movie_librarian")
        if not librarian_class:
            state["storage_result"] = {"error": "Librarian not available"}
            return state
        
        librarian = librarian_class(verbose=self.verbose)
        
        if self.verbose:
            self.log(f"[VERBOSE] Calling Librarian.store_movies({len(movies)} movies)")
        
        # Call Librarian to store the fetched movie data
        result = librarian.store_movies(movies)
        state["storage_result"] = result
        
        successful = result.get("successful", [])
        failed = result.get("failed", [])
        self.log_success(f"✓ Stored {len(successful)} movies")
        if failed:
            self.log(f"✗ Failed: {len(failed)} movies")
        
        return state
    
    def _query_movies(self, state: OrchestrationState) -> OrchestrationState:
        """Call Critic for movie recommendations."""
        self.log_success("Querying movies via Critic...")
        
        # Get Critic agent
        critic_class = AgentRegistry.get_agent("movie_critic")
        if not critic_class:
            state["result"] = {"error": "Critic not available"}
            return state
        
        critic = critic_class()
        result = critic.process(state["clean_query"])
        state["result"] = result
        
        return state
    
    def _finalize(self, state: OrchestrationState) -> OrchestrationState:
        """Finalize and prepare result."""
        if state["intent"] == "add_movie":
            # Return storage result
            state["result"] = {
                "intent": "add_movie",
                "movie_data": state.get("movie_data"),
                "storage_result": state.get("storage_result"),
                "workflow": "MovieCollector → Librarian"
            }
        elif state["intent"] == "fetch_movie":
            # Return fetch result only
            state["result"] = {
                "intent": "fetch_movie",
                "movie_data": state.get("movie_data"),
                "workflow": "MovieCollector"
            }
        # query_movie result already set in _query_movies
        
        return state

"""
Dispatcher Agent - State-based routing using LangGraph.

Uses StateGraph to explicitly model the routing decision process:
1. Analyze query
2. Classify intent
3. Select target agent
4. Extract clean query
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from agents.base_agent import BaseAgent, AgentConfig
from agents.registry import AgentRegistry


class RoutingState(TypedDict):
    """State for the routing graph."""
    query: str                  # Original user query
    query_lower: str            # Lowercase for matching
    routing_method: str         # How we routed (pattern/keyword/llm/default)
    target_agent_id: str        # Selected agent ID
    clean_query: str            # Query with prefixes removed
    confidence: float           # Routing confidence (0-1)


class Dispatcher(BaseAgent):
    """
    State-based routing agent using LangGraph.
    
    Routes requests through a StateGraph:
    analyze → pattern_match → keyword_match → select_agent → extract_query
    """
    
    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return dispatcher configuration."""
        return AgentConfig(
            agent_id="dispatcher",
            name="Dispatcher",
            description="Routes user requests to appropriate agents using state-based graph",
            patterns=[],
            keywords=[],
            capabilities=["Intent classification", "Dynamic routing", "State tracking"],
            example_queries=[],
            requires_llm=False
        )
    
    def __init__(self, model: str | None = None):
        """Initialize the Dispatcher."""
        super().__init__(model)
        self.agent_configs = AgentRegistry.get_configs()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the routing state graph."""
        workflow = StateGraph(RoutingState)
        
        # Add nodes
        workflow.add_node("analyze", self._analyze_query)
        workflow.add_node("pattern_match", self._pattern_match)
        workflow.add_node("keyword_match", self._keyword_match)
        workflow.add_node("select_agent", self._select_agent)
        workflow.add_node("extract_query", self._extract_query)
        
        # Define flow
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "pattern_match")
        
        # Conditional: If pattern matched, go to extract_query, else try keywords
        workflow.add_conditional_edges(
            "pattern_match",
            lambda state: "found" if state.get("target_agent_id") else "not_found",
            {
                "found": "extract_query",
                "not_found": "keyword_match"
            }
        )
        
        # Conditional: If keyword matched, go to extract_query, else select default
        workflow.add_conditional_edges(
            "keyword_match",
            lambda state: "found" if state.get("target_agent_id") else "not_found",
            {
                "found": "extract_query",
                "not_found": "select_agent"
            }
        )
        
        workflow.add_edge("select_agent", "extract_query")
        workflow.add_edge("extract_query", END)
        
        return workflow.compile()
    
    def process(self, query: str) -> dict:
        """
        Route a query using the state graph.
        
        Args:
            query: User's request
            
        Returns:
            Dictionary with routing decision
        """
        # Initialize state
        initial_state: RoutingState = {
            "query": query,
            "query_lower": query.lower().strip(),
            "routing_method": "unknown",
            "target_agent_id": "",
            "clean_query": query,
            "confidence": 0.0
        }
        
        # Execute graph
        result = self.graph.invoke(initial_state)
        
        # Log routing decision
        config = self.agent_configs.get(result["target_agent_id"])
        if config:
            self.log_success(f"Routed to: {config.name} ({result['routing_method']}, confidence: {result['confidence']:.2f})")
        
        return {
            "agent_id": result["target_agent_id"],
            "query": result["clean_query"],
            "routing_method": result["routing_method"],
            "confidence": result["confidence"]
        }
    
    def _analyze_query(self, state: RoutingState) -> RoutingState:
        """Analyze the query (preprocessing)."""
        # Just normalize - already done in initial state
        return state
    
    def _pattern_match(self, state: RoutingState) -> RoutingState:
        """Try pattern-based routing."""
        query_lower = state["query_lower"]
        
        for agent_id, config in self.agent_configs.items():
            for pattern in config.patterns:
                if query_lower.startswith(pattern):
                    state["target_agent_id"] = agent_id
                    state["routing_method"] = "pattern"
                    state["confidence"] = 1.0  # Pattern match is very confident
                    self.log(f"Pattern match: '{pattern}' → {config.name}")
                    return state
        
        # No pattern match
        return state
    
    def _keyword_match(self, state: RoutingState) -> RoutingState:
        """Try keyword-based routing."""
        query_lower = state["query_lower"]
        
        best_agent = None
        best_score = 0
        
        for agent_id, config in self.agent_configs.items():
            score = sum(1 for keyword in config.keywords if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        if best_agent and best_score > 0:
            config = self.agent_configs[best_agent]
            state["target_agent_id"] = best_agent
            state["routing_method"] = "keyword"
            state["confidence"] = min(best_score / 3.0, 1.0)  # Normalize score
            self.log(f"Keyword match: score={best_score} → {config.name}")
            return state
        
        # No keyword match
        return state
    
    def _select_agent(self, state: RoutingState) -> RoutingState:
        """Select default agent when no match found."""
        # Default to critic for movie queries
        default_agent = "movie_critic"
        config = self.agent_configs.get(default_agent)
        
        state["target_agent_id"] = default_agent
        state["routing_method"] = "default"
        state["confidence"] = 0.3  # Low confidence - just a guess
        
        if config:
            self.log(f"No match found, defaulting to: {config.name}")
        
        return state
    
    def _extract_query(self, state: RoutingState) -> RoutingState:
        """Extract clean query by removing routing prefixes."""
        agent_id = state["target_agent_id"]
        query = state["query"]
        query_lower = state["query_lower"]
        
        config = self.agent_configs.get(agent_id)
        if not config:
            state["clean_query"] = query.strip()
            return state
        
        # Try to remove pattern prefixes
        for pattern in config.patterns:
            if query_lower.startswith(pattern):
                state["clean_query"] = query[len(pattern):].strip()
                return state
        
        # No prefix to remove
        state["clean_query"] = query.strip()
        return state

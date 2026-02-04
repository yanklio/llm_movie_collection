"""
Dispatcher Agent - Dynamic intent routing using agent registry.

Automatically routes requests to appropriate agents based on their
self-described patterns and keywords.
"""

from agents.base_agent import BaseAgent, AgentConfig
from agents.registry import AgentRegistry


class Dispatcher(BaseAgent):
    """
    Dynamic routing agent that uses the agent registry.
    
    Routes based on:
    1. Deterministic patterns (e.g., "add:", "find:")
    2. Keyword heuristics
    3. Optional LLM classification (future)
    """
    
    @classmethod
    def get_config(cls) -> AgentConfig:
        """Return dispatcher configuration."""
        return AgentConfig(
            agent_id="dispatcher",
            name="Dispatcher",
            description="Routes user requests to appropriate agents",
            patterns=[],
            keywords=[],
            capabilities=["Intent classification", "Dynamic routing"],
            example_queries=[],
            requires_llm=False  # Can work without LLM using patterns
        )
    
    def __init__(self, model: str | None = None):
        """Initialize the Dispatcher."""
        super().__init__(model)
        self.agent_configs = AgentRegistry.get_configs()
    
    def process(self, query: str) -> dict:
        """
        Route a query to the appropriate agent.
        
        Args:
            query: User's request
            
        Returns:
            Dictionary with routing decision
        """
        agent_id = self.route(query)
        clean_query = self.extract_query(query, agent_id)
        
        return {
            "agent_id": agent_id,
            "query": clean_query,
            "routing_method": "pattern"  # or "keyword" or "llm"
        }
    
    def route(self, user_input: str) -> str | None:
        """
        Determine which agent should handle the request.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Agent ID or None if no match
        """
        user_lower = user_input.lower().strip()
        
        # 1. Pattern-based routing (highest priority)
        for agent_id, config in self.agent_configs.items():
            for pattern in config.patterns:
                if user_lower.startswith(pattern):
                    self.log_success(f"Pattern match → {config.name} ('{pattern}')")
                    return agent_id
        
        # 2. Keyword-based routing
        best_match = None
        best_score = 0
        
        for agent_id, config in self.agent_configs.items():
            score = sum(1 for keyword in config.keywords if keyword in user_lower)
            if score > best_score:
                best_score = score
                best_match = agent_id
        
        if best_match and best_score > 0:
            config = self.agent_configs[best_match]
            self.log_success(f"Keyword match → {config.name} (score: {best_score})")
            return best_match
        
        # 3. LLM-based routing (future enhancement)
        if self.llm:
            return self._llm_route(user_input)
        
        # 4. Default fallback
        self.log_error("No clear match, defaulting to critic")
        return "movie_critic"  # Default to query agent
    
    def extract_query(self, user_input: str, agent_id: str | None = None) -> str:
        """
        Remove routing prefixes from the query.
        
        Args:
            user_input: Raw user input
            agent_id: Target agent ID
            
        Returns:
            Clean query
        """
        if not agent_id:
            return user_input.strip()
        
        config = self.agent_configs.get(agent_id)
        if not config:
            return user_input.strip()
        
        user_lower = user_input.lower().strip()
        
        # Remove pattern prefixes
        for pattern in config.patterns:
            if user_lower.startswith(pattern):
                return user_input[len(pattern):].strip()
        
        return user_input.strip()
    
    def _llm_route(self, user_input: str) -> str:
        """
        Use LLM for routing (future enhancement).
        
        This could be expanded to handle ambiguous cases.
        """
        # TODO: Implement LLM-based routing
        return "movie_critic"

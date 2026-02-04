"""
Agent Registry - Centralized agent discovery and management.

This allows the system to be extended with new agents without modifying
the dispatcher logic.
"""

from typing import Type, Dict
from agents.base_agent import BaseAgent, AgentConfig
from agents.movie_collector import MovieCollector
from agents.librarian import Librarian
from agents.critic import Critic


class AgentRegistry:
    """
    Central registry for all agents in the system.
    
    New agents can be registered here to be automatically discovered
    by the dispatcher.
    """
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, agent_class: Type[BaseAgent]):
        """Register an agent class."""
        config = agent_class.get_config()
        cls._agents[config.agent_id] = agent_class
    
    @classmethod
    def get_all_agents(cls) -> Dict[str, Type[BaseAgent]]:
        """Get all registered agents."""
        return cls._agents.copy()
    
    @classmethod
    def get_agent(cls, agent_id: str) -> Type[BaseAgent] | None:
        """Get a specific agent by ID."""
        return cls._agents.get(agent_id)
    
    @classmethod
    def get_configs(cls) -> Dict[str, AgentConfig]:
        """Get all agent configurations."""
        return {
            agent_id: agent_class.get_config()
            for agent_id, agent_class in cls._agents.items()
        }

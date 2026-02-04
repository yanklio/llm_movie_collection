from agents.movie_collector import MovieCollector
from agents.librarian import Librarian
from agents.critic import Critic
from agents.registry import AgentRegistry

AgentRegistry.register(MovieCollector)
AgentRegistry.register(Librarian)
AgentRegistry.register(Critic)

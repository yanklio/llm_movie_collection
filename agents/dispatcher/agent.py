import re
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.base_agent import AgentConfig, BaseAgent
from agents.registry import AgentRegistry

from .prompts import routing_prompt


class RoutingState(TypedDict):
    """State for agent routing."""

    query: str
    target_agent: str
    clean_query: str
    routing_method: str
    result: Optional[dict]


class Dispatcher(BaseAgent):
    """
    Simple routing agent - matches patterns and routes to appropriate agents.

    No complex intent classification, just pattern matching with LLM fallback.
    """

    @classmethod
    def get_config(cls) -> AgentConfig:
        return AgentConfig(
            agent_id="dispatcher",
            name="Dispatcher",
            description="Routes queries to appropriate agents based on patterns",
            patterns=[],
            capabilities=["Pattern matching", "Agent routing"],
            example_queries=[],
        )

    def __init__(self, model: str | None = None, verbose: bool = False):
        super().__init__(model, verbose)
        self.graph = self._build_graph()

    def process(self, query: str) -> dict:
        """Route query to appropriate agent."""
        initial_state: RoutingState = {
            "query": query,
            "target_agent": "",
            "clean_query": query,
            "routing_method": "",
            "result": None,
        }

        result_state = self.graph.invoke(initial_state)
        return result_state.get("result", {})

    def _build_graph(self) -> StateGraph:
        """Build simple routing workflow."""
        workflow = StateGraph(RoutingState)

        workflow.add_node("route", self._route_query)
        workflow.add_node("execute", self._execute_agent)

        workflow.set_entry_point("route")
        workflow.add_edge("route", "execute")
        workflow.add_edge("execute", END)

        return workflow.compile()

    def _route_query(self, state: RoutingState) -> RoutingState:
        """Route query to agent based on patterns."""
        query = state["query"]

        agent_id, clean_query = self._match_patterns(query)
        if agent_id:
            state.update(
                {
                    "target_agent": agent_id,
                    "clean_query": clean_query,
                    "routing_method": "pattern",
                }
            )
            self.log(f"Pattern matched → {agent_id}")
            return state

        agent_id, clean_query = self._route_with_llm(query)
        state.update(
            {
                "target_agent": agent_id,
                "clean_query": clean_query,
                "routing_method": "llm",
            }
        )
        self.log(f"LLM routed → {agent_id}")
        return state

    def _match_patterns(self, query: str) -> tuple[str, str]:
        """Try to match query against agent patterns."""
        query_lower = query.lower().strip()

        for agent_config in AgentRegistry.get_configs().values():
            if agent_config.agent_id == "dispatcher":
                continue

            for pattern in agent_config.patterns:
                pattern_clean = pattern.rstrip(":").strip()
                if query_lower.startswith(f"{pattern_clean}:") or query_lower.startswith(
                    f"{pattern_clean} "
                ):
                    clean_query = query[len(pattern_clean) + 1 :].strip()
                    return agent_config.agent_id, clean_query

        return "", query

    def _route_with_llm(self, query: str) -> tuple[str, str]:
        """Use LLM to determine which agent to route to."""
        prompt = self._build_routing_prompt(query)

        try:
            response = self.llm.invoke(prompt)
            return self._parse_routing_response(response.content.strip(), query)
        except Exception as e:
            self.log_error(f"LLM routing failed: {e}")
            return "movie_critic", query

    def _build_routing_prompt(self, query: str) -> str:
        """Build prompt for LLM routing."""
        agents = []
        examples = []

        for agent_config in AgentRegistry.get_configs().values():
            if agent_config.agent_id == "dispatcher":
                continue

            agents.append(f"- {agent_config.agent_id}: {agent_config.description}")

            if agent_config.example_queries:
                example = agent_config.example_queries[0]
                examples.append(f'"{example}" → {agent_config.agent_id}')

        agents_text = "\n".join(agents)
        examples_text = "\n".join(examples[:3])
        agent_ids = [
            config.agent_id
            for config in AgentRegistry.get_configs().values()
            if config.agent_id != "dispatcher"
        ]

        return routing_prompt(query, agents_text, examples_text, agent_ids)

    def _parse_routing_response(self, content: str, fallback_query: str) -> tuple[str, str]:
        """Parse LLM routing response."""
        json_match = re.search(r"\{[^}]+\}", content)
        if not json_match:
            return "movie_critic", fallback_query

        try:
            import json

            result = json.loads(json_match.group())
            agent_id = result.get("agent", "movie_critic")
            clean_query = result.get("clean_query", fallback_query)
            return agent_id, clean_query
        except:
            return "movie_critic", fallback_query

    def _execute_agent(self, state: RoutingState) -> RoutingState:
        """Execute the target agent."""
        agent_id = state["target_agent"]
        clean_query = state["clean_query"]

        agent_class = AgentRegistry.get_agent(agent_id)
        if not agent_class:
            state["result"] = {"error": f"Agent '{agent_id}' not found"}
            return state

        try:
            agent = agent_class(verbose=self.verbose)
            result = agent.process(clean_query)

            if isinstance(result, dict):
                result["routed_to"] = agent_id
                result["routing_method"] = state["routing_method"]

            state["result"] = result
            self.log_success(f"Executed {agent_id}")

        except Exception as e:
            self.log_error(f"Error executing {agent_id}: {e}")
            state["result"] = {"error": f"Execution failed: {str(e)}"}

        return state

import re
from typing import Any

from utils.parsing import extract_json_from_response

from agents.base_agent import AgentConfig, BaseAgent
from agents.registry import AgentRegistry

from .prompts import next_step_prompt, routing_prompt


class Dispatcher(BaseAgent):
    """
    Simple LLM router - no tool-calling, just agent selection.
    """

    MAX_ITERATIONS = 3

    @classmethod
    def get_config(cls) -> AgentConfig:
        return AgentConfig(
            agent_id="dispatcher",
            name="Dispatcher",
            description="Routes queries to appropriate agents",
            patterns=[],
            capabilities=["Agent routing", "Context passing"],
            example_queries=[],
        )

    def __init__(self, model: str | None = None, verbose: bool = False):
        super().__init__(model, verbose)

    def process(self, query: str, chat_history: list[dict] = None) -> dict:
        """Route query through agents until complete."""
        self.log(f"Query: {query}")
        
        # State tracking
        state = {
            "query": query,
            "chat_history": chat_history,
            "context": {},
            "visited_agents": set(),
            "last_result": {},
            "iterations": 0
        }

        while state["iterations"] < self.MAX_ITERATIONS:
            # 1. Decide next action
            decision = self._get_routing_decision(state)
            
            # 2. Check completion
            if decision.get("complete"):
                self.log_success(f"Completed in {state['iterations'] + 1} step(s)")
                return {
                    "response": decision.get("response", state["last_result"].get("response", "")),
                    "success": True,
                }

            agent_id = decision.get("next_agent")
            if not agent_id or agent_id in state["visited_agents"]:
                break  # Stop if no agent or loop detected

            # 3. Execute agent
            self.log(f"→ {agent_id}")
            state["visited_agents"].add(agent_id)
            result = self._execute_agent(agent_id, decision.get("query", state["query"]), state["context"])
            
            # 4. Update state with result
            state["last_result"] = result
            state["iterations"] += 1
            
            # 5. Handle data passing (Librarian storage pattern)
            self._update_context(state, agent_id, result)
            
            # Special case: automatic completion if storage occurred
            if result.get("stored_count", 0) > 0 or result.get("skipped_count", 0) > 0:
                self.log_success(f"Storage complete in {state['iterations']} step(s)")
                return {
                    "response": result.get("message", "Operation completed"),
                    "success": True,
                    **result
                }

        return state["last_result"] if state["last_result"] else {"error": "No result", "success": False}

    def _get_routing_decision(self, state: dict) -> dict:
        """Ask LLM for the next step."""
        agents_text = self._get_agents_text()
        if state["iterations"] == 0:
            prompt = routing_prompt(state["query"], agents_text, state["chat_history"])
            self.log_model("Deciding which agent to call")
        else:
            prompt = next_step_prompt(
                state["query"], 
                state["last_result"], 
                agents_text, 
                list(state["visited_agents"]), 
                state["chat_history"]
            )
            self.log_model("Deciding next step")

        with self.thinking("Routing"):
            response = self.llm.invoke(prompt)
        return extract_json_from_response(response.content)

    def _execute_agent(self, agent_id: str, query: str, context: dict) -> dict:
        """Instantiate and run the selected agent."""
        agent_class = AgentRegistry.get_agent(agent_id)
        if not agent_class:
            return {"error": f"Agent {agent_id} not found", "success": False}

        agent = agent_class(verbose=self.verbose)
        
        # Check if we should store data instead of processing query
        if context.get("data") and hasattr(agent, "store_items"):
            result = agent.store_items(context["data"])
            context["data"] = None # Clear data after storage
            return result
            
        return agent.process(query)

    def _update_context(self, state: dict, agent_id: str, result: dict):
        """Extract data from result to pass to next agent."""
        # Check if agent returned items that need storage
        for key in ["items", "data", "results"]:
            if key in result and result[key]:
                state["context"]["data"] = result[key]
                break

    def _get_agents_text(self) -> str:
        """Get agent list for prompt."""
        lines = []
        for config in AgentRegistry.get_configs().values():
            if config.agent_id != "dispatcher":
                lines.append(f"- {config.agent_id}: {config.description}")
        return "\n".join(lines)

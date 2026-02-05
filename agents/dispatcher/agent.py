"""
Dispatcher Agent - Simple LLM-based router.

Uses LLM to decide which agent to call, but without tool-calling.
Just provides agent list in context and LLM returns next agent ID.
"""

import json
import re
from typing import Any

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

    def process(self, query: str) -> dict:
        """Route query through agents until complete."""
        self.log(f"Query: {query}")

        context: dict[str, Any] = {}
        agents_text = self._get_agents_text()
        visited_agents: set[str] = set()
        last_result: dict[str, Any] = {}

        for i in range(self.MAX_ITERATIONS):
            if i == 0:
                prompt = routing_prompt(query, agents_text)
                self.log_model("Deciding which agent to call")
            else:
                prompt = next_step_prompt(query, last_result, agents_text, list(visited_agents))
                self.log_model("Deciding next step")

            with self.thinking("Routing"):
                response = self.llm.invoke(prompt)
            decision = self._parse_json(response.content)

            if decision.get("complete"):
                self.log_success(f"Completed in {i + 1} step(s)")
                return {
                    "response": decision.get("response", last_result.get("response", "")),
                    "success": True,
                }

            agent_id = decision.get("next_agent")
            if not agent_id:
                break

            if agent_id in visited_agents:
                self.log(f"Agent {agent_id} already visited, completing workflow")
                return {
                    "response": last_result.get("response")
                    or last_result.get("message", "Operation completed"),
                    "success": True,
                }

            visited_agents.add(agent_id)
            self.log(f"→ {agent_id}")

            agent_class = AgentRegistry.get_agent(agent_id)
            if not agent_class:
                return {"error": f"Agent {agent_id} not found", "success": False}

            agent = agent_class(verbose=self.verbose)

            if context.get("data") and hasattr(agent, "store_items"):
                last_result = agent.store_items(context["data"])
                context["data"] = None

                if (
                    last_result.get("stored_count", 0) > 0
                    or last_result.get("skipped_count", 0) > 0
                ):
                    self.log_success(f"Storage complete in {i + 1} step(s)")
                    return {
                        "response": last_result.get("message", "Movies added to watchlist"),
                        "success": True,
                        **last_result,
                    }
            else:
                last_result = agent.process(decision.get("query", query))
                
                # Special case: If movie_critic returns a response, we are done
                if agent_id == "movie_critic" and (last_result.get("response") or last_result.get("movies")):
                    self.log_success(f"Movie Critic completed in {i + 1} step(s)")
                    return {
                        "response": last_result.get("response", ""),
                        "success": True,
                        **last_result
                    }

                for key in ["items", "data", "results"]:
                    if key in last_result and last_result[key]:
                        context["data"] = last_result[key]
                        break

        return last_result if last_result else {"error": "No result", "success": False}

    def _get_agents_text(self) -> str:
        """Get agent list for prompt."""
        lines = []
        for config in AgentRegistry.get_configs().values():
            if config.agent_id != "dispatcher":
                lines.append(f"- {config.agent_id}: {config.description}")
        return "\n".join(lines)

    def _parse_json(self, content) -> dict:
        """Parse JSON from LLM response."""
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
                else:
                    texts.append(str(part))
            content = " ".join(texts)
        if not isinstance(content, str):
            content = str(content)

        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"```\w*\n?", "", content).strip()
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {}

import json
import re
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.base_agent import AgentConfig, BaseAgent
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
                "Workflow coordination",
            ],
            example_queries=[],
        )

    def __init__(self, model: str | None = None, verbose: bool = False):
        """Initialize the Dispatcher."""
        super().__init__(model, verbose)
        self.agent_configs = AgentRegistry.get_configs()
        self.graph = self._build_graph()

    def _generate_intent_classification_prompt(self, query: str) -> str:
        """Generate intent classification prompt dynamically from agent registry."""
        agent_descriptions = []
        agent_examples = []
        all_agent_ids = []

        for agent_config in AgentRegistry.get_configs().values():
            if agent_config.agent_id == "dispatcher":
                continue

            all_agent_ids.append(agent_config.agent_id)

            patterns_text = ", ".join([f'"{p}"' for p in agent_config.patterns[:3]])
            keywords_text = ", ".join([f'"{k}"' for k in agent_config.keywords[:3]])
            examples_text = ", ".join([f'"{ex}"' for ex in agent_config.example_queries[:2]])

            agent_desc = f"- {agent_config.agent_id}: {agent_config.description}"
            if agent_config.patterns:
                agent_desc += f" (patterns: {patterns_text})"
            if examples_text:
                agent_desc += f" (examples: {examples_text})"

            agent_descriptions.append(agent_desc)

            if agent_config.example_queries:
                example = agent_config.example_queries[0]
                clean_query = example
                for pattern in agent_config.patterns + agent_config.keywords:
                    pattern_clean = pattern.rstrip(":").strip()
                    clean_query = clean_query.replace(pattern_clean, "").strip()

                agent_examples.append(
                    f'- "{example}" → {{"intent": "{agent_config.agent_id}", "clean_query": "{clean_query}", "confidence": 0.9}}'
                )

        agents_section = "\n".join(agent_descriptions)
        intent_options = " or ".join([f'"{agent_id}"' for agent_id in all_agent_ids])
        examples_section = "\n".join(agent_examples[:4])  # Limit examples

        return f"""Classify the user's intent for this movie-related query based on available agents.

User query: "{query}"

Available agents:
{agents_section}

Respond ONLY with a JSON object in this exact format:
{{
  "intent": {intent_options},
  "clean_query": "the movie title or search terms without action words",
  "confidence": 0.0 to 1.0
}}

Examples:
{examples_section}"""

    def _classify_intent_with_llm(self, query: str) -> tuple[str, str, float]:
        """
        Use LLM to classify user intent.

        Returns:
            (intent, clean_query, confidence)
        """
        prompt = self._generate_intent_classification_prompt(query)

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                result = json.loads(json_match.group())
                intent = result.get("intent", "query_movie")
                clean_query = result.get("clean_query", query)
                confidence = float(result.get("confidence", 0.7))

                if self.verbose:
                    self.log(
                        f"[VERBOSE] LLM classified: intent={intent}, confidence={confidence:.2f}"
                    )

                return intent, clean_query, confidence

        except Exception as e:
            self.log_error(f"LLM classification failed: {e}")

        return "movie_critic", query, 0.3

    def _build_graph(self) -> StateGraph:
        """Build the orchestration state graph."""
        workflow = StateGraph(OrchestrationState)

        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("route_to_agent", self._route_to_agent)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("classify_intent")

        workflow.add_edge("classify_intent", "route_to_agent")
        workflow.add_edge("route_to_agent", "finalize")
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

        # Check patterns from registered agents first (high confidence)
        for agent_config in AgentRegistry.get_configs().values():
            if agent_config.agent_id == "dispatcher":
                continue

            for pattern in agent_config.patterns:
                pattern_clean = pattern.rstrip(":").strip()
                if query_lower.startswith(pattern_clean + ":") or query_lower.startswith(
                    pattern_clean + " "
                ):
                    state["intent"] = agent_config.agent_id
                    state["target_agent_id"] = agent_config.agent_id
                    state["routing_method"] = "pattern"
                    state["confidence"] = 1.0
                    state["clean_query"] = state["query"][len(pattern_clean) + 1 :].strip()
                    self.log(f"Intent: {agent_config.agent_id} (pattern: '{pattern_clean}')")
                    return state

        self.log("No explicit pattern found, using LLM classification...")
        intent, clean_query, confidence = self._classify_intent_with_llm(state["query"])

        state["intent"] = intent
        state["target_agent_id"] = intent
        state["routing_method"] = "llm"
        state["confidence"] = confidence
        state["clean_query"] = clean_query
        self.log(f"Intent: {intent} (LLM, confidence: {confidence:.2f})")

        return state

    def _route_to_agent(self, state: OrchestrationState) -> OrchestrationState:
        """Route to the appropriate agent based on classified intent."""
        target_agent_id = state["target_agent_id"]

        self.log_success(f"Routing to agent: {target_agent_id}")

        agent_class = AgentRegistry.get_agent(target_agent_id)
        if not agent_class:
            state["result"] = {"error": f"Agent '{target_agent_id}' not available"}
            return state

        try:
            agent = agent_class(verbose=self.verbose)
            result = agent.process(state["clean_query"])
            state["result"] = result

        except Exception as e:
            self.log_error(f"Error processing with {target_agent_id}: {e}")
            state["result"] = {"error": f"Processing failed: {str(e)}"}

        return state

    def _finalize(self, state: OrchestrationState) -> OrchestrationState:
        """Finalize and prepare result."""
        # Ensure result includes routing metadata
        if state.get("result") and isinstance(state["result"], dict):
            state["result"]["routed_to"] = state["target_agent_id"]
            state["result"]["routing_method"] = state["routing_method"]
            state["result"]["confidence"] = state["confidence"]

        return state

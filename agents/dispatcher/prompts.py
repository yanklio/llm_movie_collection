"""Prompt templates for the Dispatcher agent."""


def routing_prompt(query: str, agents_text: str, examples_text: str, agent_ids: list[str]) -> str:
    """Generate LLM routing prompt."""
    return f"""Route this movie query to the best agent.

Query: "{query}"

Available agents:
{agents_text}

Return JSON:
{{
  "agent": "{'" | "'.join(agent_ids)}",
  "clean_query": "cleaned query text"
}}

Examples:
{examples_text}"""

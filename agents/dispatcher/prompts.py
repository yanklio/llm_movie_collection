def routing_prompt(query: str, agents_text: str, chat_history: list[dict] = None) -> str:
    """Initial routing prompt."""
    history_text = _format_history(chat_history)
    
    return f"""Route this query to the best agent based on their capabilities.

Conversation History:
{history_text}

Query: "{query}"

Available agents:
{agents_text}

ROUTING INSTRUCTIONS:
- Analyze the user's intent and match it to the most suitable agent's description.
- Use "scout" ONLY for fetching NEW data from external sources (internet, APIs).
- Use "librarian" for managing or querying the EXISTING local collection/watchlist.
- Use "movie_critic" for subjective queries, recommendations, or qualitative analysis.

Return JSON:
{{"next_agent": "<agent_id>", "query": "<query for agent>"}}"""


def next_step_prompt(original_query: str, last_result: dict, agents_text: str, visited_agents: list[str], chat_history: list[dict] = None) -> str:
    """Decide next step based on result indicators."""
    indicators = []
    
    if last_result.get("stored_count"):
        indicators.append(f"stored_count: {last_result['stored_count']}")
    if last_result.get("skipped_count"):
        indicators.append(f"skipped_count: {last_result['skipped_count']}")
    if last_result.get("count"):
        indicators.append(f"count: {last_result['count']}")
    if last_result.get("deleted_count"):
        indicators.append(f"deleted_count: {last_result['deleted_count']}") 
    if last_result.get("response"):
        resp = str(last_result['response'])
        snippet = resp[:300] + "..." if len(resp) > 300 else resp
        indicators.append(f"response: {snippet}")
    if last_result.get("error"):
        indicators.append(f"error: {last_result['error']}")
    
    has_data = any(k in last_result for k in ["items", "data", "results"])
    if has_data:
        indicators.append("DATA_FETCHED: true - route to librarian to store")
    
    visited_text = ", ".join(visited_agents) if visited_agents else "none"
    history_text = _format_history(chat_history)
    
    return f"""Conversation History:
{history_text}

Original query: "{original_query}"

Result from last agent: {', '.join(indicators) if indicators else 'completed'}
Already visited: {visited_text}

Available agents:
{agents_text}

COMPLETION RULES:
1. If the response fully answers the user's question -> COMPLETE
2. If data was stored or actions completed successfully -> COMPLETE
3. If new data was fetched (DATA_FETCHED: true), route to 'librarian' to store it.
4. Do NOT loop back to the same agent repeatedly.

Return JSON:
- Complete: {{"complete": true, "response": "<final answer summary>"}}
- Need another agent: {{"complete": false, "next_agent": "<agent_id>", "query": "<query>"}}

STRICT FORMATTING RULES (you MUST follow these):
- PLAIN TEXT ONLY - no markdown
- NO tables (use bullet lists instead)
- NO bold (**) or italic (*) formatting
- Use simple dash (-) for lists"""

def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "None"
    
    lines = []
    for msg in history[-5:]:  # Only show last 5 turns
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\\n".join(lines)

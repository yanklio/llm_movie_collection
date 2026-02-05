def routing_prompt(query: str, agents_text: str) -> str:
    """Initial routing prompt."""
    return f"""Route this query to the best agent.

Query: "{query}"

Available agents:
{agents_text}

Return JSON:
{{"next_agent": "<agent_id>", "query": "<query for agent>"}}"""


def next_step_prompt(original_query: str, last_result: dict, agents_text: str, visited_agents: list[str]) -> str:
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
        snippet = resp[:200] + "..." if len(resp) > 200 else resp
        indicators.append(f"response: {snippet}")
    if last_result.get("error"):
        indicators.append(f"error: {last_result['error']}")
    
    has_data = any(k in last_result for k in ["items", "data", "results"])
    if has_data:
        indicators.append("DATA_FETCHED: true - MUST route to storage agent next")
    
    visited_text = ", ".join(visited_agents) if visited_agents else "none"
    
    return f"""Original query: "{original_query}"

Result: {', '.join(indicators) if indicators else 'completed'}
Already visited agents: {visited_text}

Available agents:
{agents_text}

RULES:
1. If stored_count > 0 or skipped_count > 0: Task is COMPLETE - return {{"complete": true, "response": "..."}}
2. If DATA_FETCHED is true: Route to movie_librarian to store the data
3. NEVER call an agent that was already visited
4. If response contains the answer: Task is COMPLETE

Return JSON:
- If task complete: {{"complete": true, "response": "<final response>"}}
- If need another agent: {{"complete": false, "next_agent": "<agent_id>", "query": "<query>"}}"""


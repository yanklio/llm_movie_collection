def routing_prompt(query: str, agents_text: str) -> str:
    """Initial routing prompt."""
    return f"""Route this query to the best agent.

Query: "{query}"

Available agents:
{agents_text}

ROUTING RULES:
- "check", "do I have", "in my collection/watchlist", "show me my" → librarian (searches user's collection)
- "add", "save", "fetch" (getting NEW content from internet) → scout
- "recommend", "suggest", "find something like" (personalized recommendations) → movie_critic

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
        snippet = resp[:300] + "..." if len(resp) > 300 else resp
        indicators.append(f"response: {snippet}")
    if last_result.get("error"):
        indicators.append(f"error: {last_result['error']}")
    
    has_data = any(k in last_result for k in ["items", "data", "results"])
    if has_data:
        indicators.append("DATA_FETCHED: true - route to librarian to store")
    
    visited_text = ", ".join(visited_agents) if visited_agents else "none"
    
    return f"""Original query: "{original_query}"

Result from last agent: {', '.join(indicators) if indicators else 'completed'}
Already visited: {visited_text}

Available agents:
{agents_text}

COMPLETION RULES:
1. If response answers the user's question → COMPLETE
2. If stored_count > 0 or skipped_count > 0 → COMPLETE
3. If DATA_FETCHED is true → route to librarian
4. NEVER call an already-visited agent

ROUTING HINTS:
- If Librarian found items -> return info about the entities from user collection
- NEVER use internet search agent (scout) for items that Librarian just found (Scout is ONLY for adding NEW content from internet)

Return JSON:
- Complete: {{"complete": true, "response": "<format the result nicely>"}}
- Need another agent: {{"complete": false, "next_agent": "<agent_id>", "query": "<query>"}}

STRICT FORMATTING RULES (you MUST follow these):
- PLAIN TEXT ONLY - no markdown
- NO tables (use bullet lists instead)
- NO bold (**) or italic (*) formatting
- Use simple dash (-) for lists"""

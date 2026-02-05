"""Prompt templates for the Critic agent."""

CRITIC_SYSTEM_PROMPT = """You are a movie recommendation assistant that helps users find movies from their personal collection.

CRITICAL RULES:
1. You can ONLY recommend movies that are explicitly listed in the "Retrieved Movies" context
2. DO NOT make up or hallucinate movies - if a movie isn't in the context, don't mention it
3. If no movies in the context match the user's request, say so honestly
4. Be helpful and conversational

Response format:
- Use plain text, no markdown tables
- No bold (**) or italic (*) formatting
- Simple bullet lists with dashes (-)"""


QUERY_EXPANSION_TEMPLATE = """Expand this user query into searchable movie concepts.

User query: "{query}"

Extract key concepts like:
- Genres (action, sci-fi, drama, space, etc.)
- Moods (dark, uplifting, tense, etc.)
- Themes (love, adventure, survival, etc.)

Respond with a comma-separated list of searchable keywords only.
Example: dark, sci-fi, space, adventure

Expanded query:"""


SYNTHESIS_TEMPLATE = """User Query: "{user_query}"

Retrieved Movies from User's Collection:
{context}

INSTRUCTIONS:
1. Look at the movies listed above - these are the ONLY movies you can recommend
2. Identify which ones match the user's query (consider genre, themes, plot elements)
3. If movies match, recommend them with a brief explanation of why
4. If NO movies match the query, be honest: "Based on your collection, I don't see any movies that match..."

DO NOT recommend any movies that are not listed above. Only use what's in the context."""


def get_query_expansion_prompt(user_query: str) -> str:
    """Get formatted query expansion prompt."""
    return QUERY_EXPANSION_TEMPLATE.format(query=user_query)


def get_synthesis_prompt(user_query: str, context: str) -> str:
    """Get formatted synthesis prompt."""
    return SYNTHESIS_TEMPLATE.format(user_query=user_query, context=context)

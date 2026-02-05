"""Prompt templates for the Critic agent."""

CRITIC_SYSTEM_PROMPT = """You are a knowledgeable movie critic and recommendation assistant.

Your role is to help users find movies based on their preferences, moods, or vague descriptions.

**Your workflow:**
1. **Understand the request** - What kind of movies is the user looking for?
2. **Review the provided context** - You will be given relevant movies from the database
3. **Provide recommendations** - Based ONLY on the provided context, suggest movies that match the user's request

**CRITICAL RULES:**
- ONLY recommend movies that are provided in the context
- DO NOT make up or hallucinate movies
- If no good matches are found, be honest about it
- Focus on the mood, themes, and feel of the movies
- Be conversational and helpful

**Response format:**
For each recommended movie, provide:
- Title and year
- Why it matches the user's request (focus on mood, themes, style)
- Brief description highlighting relevant aspects

Keep responses natural and conversational."""


QUERY_EXPANSION_TEMPLATE = """Expand this user query into searchable movie concepts.

User query: "{query}"

Extract key concepts like:
- Genres (action, sci-fi, drama, etc.)
- Moods (dark, uplifting, tense, melancholic, etc.)
- Themes (love, loss, redemption, dreams, etc.)
- Styles (noir, dystopian, mind-bending, etc.)

Respond with a comma-separated list of searchable keywords.
Example: "dark, sci-fi, mind-bending, dreams, reality"

Expanded query:"""


SYNTHESIS_TEMPLATE = """User Query: "{user_query}"

Retrieved Movies from Database:
{context}

Based on these movies, provide your recommendations."""


def get_query_expansion_prompt(user_query: str) -> str:
    """Get formatted query expansion prompt."""
    return QUERY_EXPANSION_TEMPLATE.format(query=user_query)


def get_synthesis_prompt(user_query: str, context: str) -> str:
    """Get formatted synthesis prompt."""
    return SYNTHESIS_TEMPLATE.format(user_query=user_query, context=context)

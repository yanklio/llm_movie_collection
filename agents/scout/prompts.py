"""Prompt templates for the Movie Collector agent."""

QUERY_ANALYSIS_PROMPT = """You are a movie search query analyzer. Your task is to analyze user queries and determine the best search strategy.

Analyze this query and determine:
1. What type of search to perform (person, title, or keyword)
2. Extract relevant search terms
3. Provide confidence score

Query types:
- PERSON: When user asks about specific actors, directors, or filmmakers
- TITLE: When user mentions a specific movie title
- KEYWORD: When user describes themes, genres, or general concepts

User query: "{query}"

Respond in JSON format:
{{
    "search_type": "person|title|keyword",
    "search_terms": "cleaned search terms",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Examples:
- "Brad Pitt movies" → {{"search_type": "person", "search_terms": "Brad Pitt", "confidence": 0.9}}
- "The Matrix" → {{"search_type": "title", "search_terms": "The Matrix", "confidence": 0.95}}
- "dark sci-fi thriller" → {{"search_type": "keyword", "search_terms": "dark sci-fi thriller", "confidence": 0.8}}"""


RESULT_SUMMARIZATION_PROMPT = """You are a movie information summarizer. Create a concise, informative summary of the search results.

Search Query: "{query}"
Search Method: {search_method}
Results Found: {result_count}

Movie Data:
{movie_data}

Create a brief, conversational summary that:
1. Acknowledges what the user was looking for
2. Summarizes the key findings
3. Highlights interesting details if relevant
4. Uses a friendly, helpful tone

Keep it concise but informative."""


def get_query_analysis_prompt(query: str) -> str:
    """Get formatted query analysis prompt."""
    return QUERY_ANALYSIS_PROMPT.format(query=query)


def get_result_summary_prompt(
    query: str, search_method: str, result_count: int, movie_data: str
) -> str:
    """Get formatted result summarization prompt."""
    return RESULT_SUMMARIZATION_PROMPT.format(
        query=query, search_method=search_method, result_count=result_count, movie_data=movie_data
    )

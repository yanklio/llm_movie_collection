"""Prompt templates for the Librarian agent."""

MOVIE_SUMMARY_GENERATION_PROMPT = """You are a movie librarian creating searchable summaries for a personal movie database.

Create a rich, searchable summary that captures the essence of this movie for future retrieval.

Movie Information:
Title: {title}
Year: {year}
Genre: {genre}
Director: {director}
Cast: {actors}
Plot: {plot}
Runtime: {runtime}
Rating: {rating}/10

Your summary should:
1. Highlight key themes and moods
2. Include searchable keywords for genres, emotions, and concepts
3. Mention notable cast and crew
4. Capture the film's unique qualities and appeal
5. Be conversational yet informative

Focus on elements that would help someone find this movie later based on:
- Mood (dark, uplifting, intense, romantic, etc.)
- Themes (redemption, family, betrayal, coming-of-age, etc.)
- Style (noir, dystopian, surreal, realistic, etc.)
- Notable performances or direction

Create a summary that's both human-readable and rich with searchable concepts."""


QUERY_UNDERSTANDING_PROMPT = """You are a librarian assistant helping to understand user requests about their movie watchlist.

Analyze this user query and determine:
1. What operation they want to perform
2. Extract the movie title if mentioned
3. Understand the intent clearly

User query: "{query}"

Operations:
- CHECK: User wants to know if a movie is in their watchlist
- DELETE: User wants to remove a movie from their watchlist
- SEARCH: User wants to find movies in their collection
- INFO: User wants details about a movie in their collection

Respond in JSON format:
{{
    "operation": "check|delete|search|info",
    "movie_title": "extracted title or null",
    "intent": "clear description of what user wants",
    "confidence": 0.0-1.0
}}

Examples:
- "Do I have Inception?" → {{"operation": "check", "movie_title": "Inception", "intent": "Check if Inception is in watchlist"}}
- "Remove The Matrix" → {{"operation": "delete", "movie_title": "The Matrix", "intent": "Delete The Matrix from watchlist"}}
- "Tell me about Pulp Fiction" → {{"operation": "info", "movie_title": "Pulp Fiction", "intent": "Get details about Pulp Fiction"}}"""


STORAGE_CONFIRMATION_PROMPT = """You are a friendly librarian confirming movie additions to a personal watchlist.

Movies successfully added: {successful_count}
Movies failed to add: {failed_count}

Successful additions:
{successful_movies}

{failed_info}

Create a warm, conversational response that:
1. Celebrates the successful additions
2. Briefly mentions what makes each movie special
3. Addresses any failures if they occurred
4. Maintains enthusiasm for the user's movie collection

Keep it personal and engaging, like a knowledgeable friend helping build their movie library."""


def get_movie_summary_prompt(movie_info: dict) -> str:
    """Get formatted movie summary generation prompt."""
    return MOVIE_SUMMARY_GENERATION_PROMPT.format(
        title=movie_info.get("title", "Unknown"),
        year=movie_info.get("year", "N/A"),
        genre=movie_info.get("genre", "N/A"),
        director=movie_info.get("director", "Unknown"),
        actors=movie_info.get("actors", "Unknown"),
        plot=movie_info.get("plot", "No plot available"),
        runtime=movie_info.get("runtime", "Unknown"),
        rating=movie_info.get("rating", "N/A"),
    )


def get_query_understanding_prompt(query: str) -> str:
    """Get formatted query understanding prompt."""
    return QUERY_UNDERSTANDING_PROMPT.format(query=query)


def get_storage_confirmation_prompt(successful_movies: list, failed_movies: list) -> str:
    """Get formatted storage confirmation prompt."""
    successful_count = len(successful_movies)
    failed_count = len(failed_movies)

    successful_info = (
        "\n".join(
            [f"• {movie.title} ({movie.year}) - {movie.genre}" for movie in successful_movies]
        )
        if successful_movies
        else "None"
    )

    failed_info = ""
    if failed_movies:
        failed_info = "\nFailed to add:\n" + "\n".join([f"• {movie}" for movie in failed_movies])

    return STORAGE_CONFIRMATION_PROMPT.format(
        successful_count=successful_count,
        failed_count=failed_count,
        successful_movies=successful_info,
        failed_info=failed_info,
    )

from typing import List

from pydantic import BaseModel


class LibrarianRequest(BaseModel):
    query: str


class WatchlistMovie(BaseModel):
    title: str
    year: str
    genre: str
    director: str
    rating: str
    id: str


class LibrarianResponse(BaseModel):
    query: str
    operation_performed: str
    movies: List[WatchlistMovie] = []
    message: str
    success: bool = True
    agent_id: str = "movie_librarian"


class MovieStorage(BaseModel):
    movies_stored: int
    movies_skipped: int
    stored_titles: List[str] = []
    message: str

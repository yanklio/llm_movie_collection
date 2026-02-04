from typing import List, Optional

from pydantic import BaseModel


class CriticRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class MovieResult(BaseModel):
    id: str
    document: str
    metadata: dict
    distance: float


class CriticResponse(BaseModel):
    query: str
    expanded_query: str
    movies: List[MovieResult]
    response: str
    agent_id: str = "movie_critic"

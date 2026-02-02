"""
MovieAPITool - External API communication for movie data retrieval.

Handles all interactions with the OMDb API to fetch movie metadata.
"""

import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv

from .movie_info import MovieInfo

load_dotenv()


class MovieAPITool:
    """
    Tool for fetching movie data from the OMDb API.
    
    Supports searching by:
    - Movie title
    - IMDb ID (e.g., tt1375666)
    - IMDb URL (e.g., https://www.imdb.com/title/tt1375666/)
    """
    
    BASE_URL = "http://www.omdbapi.com/"
    IMDB_URL_PATTERN = re.compile(r"imdb\.com/title/(tt\d+)")
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MovieAPITool.
        
        Args:
            api_key: OMDb API key. If not provided, reads from OMDB_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OMDB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OMDb API key required. Set OMDB_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def search_by_title(self, title: str, year: Optional[str] = None) -> MovieInfo:
        """Search for a movie by its title."""
        params = {
            "apikey": self.api_key,
            "t": title,
            "plot": "full",
        }
        if year:
            params["y"] = year
            
        return self._fetch_and_normalize(params)
    
    def search_by_imdb_id(self, imdb_id: str) -> MovieInfo:
        """Search for a movie by its IMDb ID."""
        params = {
            "apikey": self.api_key,
            "i": imdb_id,
            "plot": "full",
        }
        return self._fetch_and_normalize(params)
    
    def parse_imdb_url(self, url: str) -> Optional[str]:
        """Extract IMDb ID from an IMDb URL."""
        match = self.IMDB_URL_PATTERN.search(url)
        return match.group(1) if match else None
    
    def search(self, query: str) -> MovieInfo:
        """Smart search that handles titles, IMDb IDs, and URLs."""
        # Check if it's an IMDb URL
        imdb_id = self.parse_imdb_url(query)
        if imdb_id:
            return self.search_by_imdb_id(imdb_id)
        
        # Check if it's an IMDb ID directly
        if query.startswith("tt") and query[2:].isdigit():
            return self.search_by_imdb_id(query)
        
        # Otherwise, treat as title
        return self.search_by_title(query)
    
    def _fetch_and_normalize(self, params: dict) -> MovieInfo:
        """Fetch data from OMDb API and normalize to MovieInfo."""
        response = requests.get(self.BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("Response") == "False":
            error = data.get("Error", "Unknown error")
            raise ValueError(f"OMDb API error: {error}")
        
        return self._normalize(data)
    
    def _normalize(self, data: dict) -> MovieInfo:
        """Normalize OMDb API response to MovieInfo structure."""
        return MovieInfo(
            imdb_id=data.get("imdbID", ""),
            title=data.get("Title", ""),
            year=data.get("Year", ""),
            genre=data.get("Genre", ""),
            director=data.get("Director", ""),
            actors=data.get("Actors", ""),
            plot=data.get("Plot", ""),
            poster_url=data.get("Poster", ""),
            imdb_rating=data.get("imdbRating", "N/A"),
            runtime=data.get("Runtime", ""),
        )

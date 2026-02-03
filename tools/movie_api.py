"""
MovieAPITool - External API communication for movie data retrieval.

Handles all interactions with TMDB API to fetch movie metadata and person searches.
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
    Tool for fetching movie data from TMDB (The Movie Database) API.
    
    Supports searching by:
    - Movie title
    - Person/Actor name (returns their filmography)
    - IMDb ID (e.g., tt1375666)
    - IMDb URL (e.g., https://www.imdb.com/title/tt1375666/)
    """
    
    BASE_URL = "https://api.themoviedb.org/3"
    IMDB_URL_PATTERN = re.compile(r"imdb\.com/title/(tt\d+)")
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MovieAPITool.
        
        Args:
            api_key: TMDB API key. If not provided, reads from TMDB_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TMDB API key required. Set TMDB_API_KEY environment variable "
                "or pass api_key parameter. Get one at https://www.themoviedb.org/settings/api"
            )
    
    def search_by_title(self, title: str, year: Optional[str] = None) -> MovieInfo:
        """Search for a movie by its title."""
        params = {
            "api_key": self.api_key,
            "query": title,
        }
        if year:
            params["year"] = year
        
        # Search for movies matching the title
        response = requests.get(
            f"{self.BASE_URL}/search/movie",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if not results:
            raise ValueError(f"No movie found with title: {title}")
        
        # Get the first (most relevant) result
        movie_id = results[0]["id"]
        
        # Fetch full movie details
        return self._get_movie_details(movie_id)
    
    def search_by_imdb_id(self, imdb_id: str) -> MovieInfo:
        """Search for a movie by its IMDb ID."""
        params = {
            "api_key": self.api_key,
            "external_source": "imdb_id",
        }
        
        response = requests.get(
            f"{self.BASE_URL}/find/{imdb_id}",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("movie_results", [])
        if not results:
            raise ValueError(f"No movie found with IMDb ID: {imdb_id}")
        
        movie_id = results[0]["id"]
        return self._get_movie_details(movie_id)
    
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
    
    def search_movies(self, search_term: str, year: Optional[str] = None) -> list[dict]:
        """
        Search for multiple movies matching a search term.
        
        Args:
            search_term: Search query (movie title fragment)
            year: Optional year filter
            
        Returns:
            List of movie dictionaries with basic info (Title, Year, imdbID)
        """
        params = {
            "api_key": self.api_key,
            "query": search_term,
        }
        if year:
            params["year"] = year
        
        response = requests.get(
            f"{self.BASE_URL}/search/movie",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if not results:
            raise ValueError(f"No movies found matching: {search_term}")
        
        # Convert TMDB format to our format with IMDb IDs
        movies = []
        for result in results[:10]:  # Limit to 10
            # Get IMDb ID for each movie
            try:
                movie_details = self._get_movie_details(result["id"])
                movies.append({
                    "Title": result.get("title", ""),
                    "Year": result.get("release_date", "")[:4] if result.get("release_date") else "",
                    "imdbID": movie_details.imdb_id,
                    "Type": "movie",
                })
            except:
                # Skip if we can't get IMDb ID
                continue
        
        return movies
    
    def search_person(self, person_name: str) -> list[dict]:
        """
        Search for movies by person/actor name.
        
        Args:
            person_name: Actor or director name (e.g., "Tom Hanks", "Christopher Nolan")
            
        Returns:
            List of movies the person acted in or directed
        """
        # Step 1: Search for the person
        params = {
            "api_key": self.api_key,
            "query": person_name,
        }
        
        response = requests.get(
            f"{self.BASE_URL}/search/person",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if not results:
            raise ValueError(f"No person found with name: {person_name}")
        
        # Get the first (most relevant) person
        person_id = results[0]["id"]
        
        # Step 2: Get their movie credits
        response = requests.get(
            f"{self.BASE_URL}/person/{person_id}/movie_credits",
            params={"api_key": self.api_key},
            timeout=10
        )
        response.raise_for_status()
        credits_data = response.json()
        
        # Combine cast and crew movies
        cast_movies = credits_data.get("cast", [])
        crew_movies = credits_data.get("crew", [])
        
        # Convert to our format - prioritize director credits
        movies = []
        seen_ids = set()
        
        # First add movies as director
        for movie in crew_movies:
            if movie.get("job") == "Director":
                movie_id = movie.get("id")
                if movie_id and movie_id not in seen_ids:
                    seen_ids.add(movie_id)
                    try:
                        movie_details = self._get_movie_details(movie_id)
                        movies.append({
                            "Title": movie.get("title", ""),
                            "Year": movie.get("release_date", "")[:4] if movie.get("release_date") else "",
                            "imdbID": movie_details.imdb_id,
                            "Type": "movie",
                            "Role": "Director",
                        })
                    except:
                        continue
        
        # Then add cast movies (if not already added as director)
        for movie in cast_movies[:20]:  # Limit to 20 most popular
            movie_id = movie.get("id")
            if movie_id and movie_id not in seen_ids:
                seen_ids.add(movie_id)
                try:
                    movie_details = self._get_movie_details(movie_id)
                    movies.append({
                        "Title": movie.get("title", ""),
                        "Year": movie.get("release_date", "")[:4] if movie.get("release_date") else "",
                        "imdbID": movie_details.imdb_id,
                        "Type": "movie",
                        "Role": "Actor",
                    })
                except:
                    continue
        
        return movies
    
    def _get_movie_details(self, tmdb_id: int) -> MovieInfo:
        """Fetch detailed movie information from TMDB."""
        params = {
            "api_key": self.api_key,
            "append_to_response": "credits,external_ids",
        }
        
        response = requests.get(
            f"{self.BASE_URL}/movie/{tmdb_id}",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        return self._normalize(data)
    
    def _normalize(self, data: dict) -> MovieInfo:
        """Normalize TMDB API response to MovieInfo structure."""
        # Extract IMDb ID from external_ids
        external_ids = data.get("external_ids", {})
        imdb_id = external_ids.get("imdb_id", "")
        
        # Extract credits
        credits = data.get("credits", {})
        
        # Get director
        crew = credits.get("crew", [])
        directors = [person["name"] for person in crew if person.get("job") == "Director"]
        director = ", ".join(directors[:3]) if directors else "N/A"
        
        # Get cast
        cast = credits.get("cast", [])
        actors = [person["name"] for person in cast[:5]]  # Top 5 actors
        actors_str = ", ".join(actors) if actors else "N/A"
        
        # Get genres
        genres = data.get("genres", [])
        genre_str = ", ".join([g["name"] for g in genres]) if genres else "N/A"
        
        # Get year from release date
        release_date = data.get("release_date", "")
        year = release_date[:4] if release_date else "N/A"
        
        # Get rating (TMDB uses 0-10 scale like IMDb)
        rating = data.get("vote_average", "N/A")
        if rating != "N/A":
            rating = f"{float(rating):.1f}"
        
        return MovieInfo(
            imdb_id=imdb_id or f"tmdb_{data.get('id', '')}",  # Fallback to TMDB ID if no IMDb ID
            title=data.get("title", ""),
            year=year,
            genre=genre_str,
            director=director,
            actors=actors_str,
            plot=data.get("overview", ""),
            poster_url=f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get("poster_path") else "",
            imdb_rating=rating,
            runtime=f"{data.get('runtime', 0)} min" if data.get("runtime") else "N/A",
        )

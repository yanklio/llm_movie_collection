from dataclasses import dataclass


@dataclass
class MovieInfo:
    """Normalized movie info structure."""
    imdb_id: str
    title: str
    year: str
    genre: str
    director: str
    actors: str
    plot: str
    poster_url: str
    imdb_rating: str
    runtime: str
    
    def to_document(self) -> str:
        """Convert to a text document for embedding."""
        return (
            f"Title: {self.title} ({self.year})\n"
            f"Genre: {self.genre}\n"
            f"Director: {self.director}\n"
            f"Cast: {self.actors}\n"
            f"Runtime: {self.runtime}\n"
            f"Rating: {self.imdb_rating}/10\n"
            f"Plot: {self.plot}"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage metadata."""
        return {
            "imdb_id": self.imdb_id,
            "title": self.title,
            "year": self.year,
            "genre": self.genre,
            "director": self.director,
            "actors": self.actors,
            "plot": self.plot,
            "poster_url": self.poster_url,
            "imdb_rating": self.imdb_rating,
            "runtime": self.runtime,
        }


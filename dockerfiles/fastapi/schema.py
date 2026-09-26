"""
Request and response schemas for the inference API.

The fields mirror the TMDB columns used in notebooks/train.ipynb. model.build_features()
turns a Movie into the columns the model was trained on.
"""
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie to predict revenue for, described with TMDB data.

    Genres and languages not seen in training are accepted: unknown genres are ignored and
    unknown languages are treated as "otro", the same as in training.
    """

    budget: int = Field(gt=0, description="Budget in USD")
    popularity: float = Field(ge=0, description="TMDB popularity score")
    vote_count: int = Field(gt=0, description="Number of TMDB votes")
    vote_average: float = Field(gt=0, le=10, description="Average TMDB rating, from 0 to 10")
    runtime: int = Field(gt=0, description="Duration in minutes")
    release_year: int = Field(ge=1850, le=2100, description="Release year")
    original_language: str = Field(description="ISO 639-1 code, e.g. 'en'")
    genres: list[str] = Field(default=[], description="TMDB genre names, e.g. ['Drama', 'Science Fiction']")
    n_production_companies: int = Field(default=1, ge=0, description="Number of production companies")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "budget": 25000000, 
                "popularity": 16.3, 
                "vote_count": 3427, 
                "vote_average": 7.2,
                "runtime": 143, 
                "release_year": 2006, 
                "original_language": "en",
                "genres": ["Drama", "Crime"], 
                "n_production_companies": 6
                }]
        }
    }

class Prediction(BaseModel):
    """Predicted revenue and the registered model version that produced it."""

    revenue: float = Field(description="Predicted revenue in USD")
    model_name: str = Field(description="Registered model name in MLflow")
    model_version: str = Field(description="Registered model version used for the prediction")
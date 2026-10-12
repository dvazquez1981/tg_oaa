"""
Esquemas de entrada y salida de la API.

Los campos corresponden a las columnas de TMDB que usa el modelo. model.build_features()
convierte una Movie en las columnas con las que se entrenó (ver airflow/dags/utils/data.py).
"""

from pydantic import BaseModel, Field


class Movie(BaseModel):
    """Película para la que se quiere estimar la recaudación, con datos de TMDB.

    Se aceptan géneros e idiomas que el modelo no vio al entrenar: los géneros desconocidos
    se ignoran y los idiomas desconocidos se tratan como "otro", igual que en el entrenamiento.
    """

    budget: int = Field(gt=0, description="Presupuesto en USD")
    popularity: float = Field(ge=0, description="Popularidad en TMDB")
    vote_count: int = Field(gt=0, description="Cantidad de votos en TMDB")
    vote_average: float = Field(gt=0, le=10, description="Puntaje promedio en TMDB, de 0 a 10")
    runtime: int = Field(gt=0, description="Duración en minutos")
    release_year: int = Field(ge=1850, le=2100, description="Año de estreno")
    original_language: str = Field(description="Código ISO 639-1 del idioma original, ej. 'en'")
    genres: list[str] = Field(
        default=[], description="Géneros con los nombres de TMDB, ej. ['Drama', 'Science Fiction']"
    )
    n_production_companies: int = Field(
        default=1, ge=0, description="Cantidad de productoras"
    )

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
                "n_production_companies": 6,
            }]
        }
    }


class Prediction(BaseModel):
    """Recaudación estimada y la versión del modelo que la calculó."""

    revenue: float = Field(description="Recaudación estimada en USD")
    model_name: str = Field(description="Nombre del modelo en el Model Registry de MLflow")
    model_version: str = Field(description="Versión del modelo que hizo la predicción")


class Health(BaseModel):
    """Estado del servicio."""

    status: str = Field(description="'ok' si hay un modelo cargado, 'sin_modelo' si no")
    model_name: str = Field(description="Nombre del modelo en el Model Registry de MLflow")
    model_alias: str = Field(description="Alias que sigue la API, normalmente 'champion'")
    model_version: str | None = Field(description="Versión cargada, o null si no hay modelo")

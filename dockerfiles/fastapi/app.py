"""
API REST que sirve el modelo de recaudación de películas.

Documentación interactiva en http://localhost:8800/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import model
from schema import Health, Movie, Prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si todavía no hay modelo registrado, la API arranca igual y /predict devuelve 503
    model.refresh(force=True)
    yield


app = FastAPI(
    title="Recaudación de películas",
    description="Estima la recaudación de una película a partir de sus datos de TMDB. "
                "Sirve el modelo con alias champion del Model Registry de MLflow.",
    lifespan=lifespan,
)

@app.get("/")
def read_root():
    """Mensaje de bienvenida."""
    return {"message": "Servicio de predicción de recaudación. Documentación en /docs"}


@app.get("/health", response_model=Health)
def health():
    """Estado del servicio y versión del modelo cargado."""
    model.refresh()
    return Health(
        status="ok" if model.is_loaded() else "sin_modelo",
        model_name=model.MODEL_NAME,
        model_alias=model.MODEL_ALIAS,
        model_version=model.model_state["version"],
    )


@app.post("/predict", response_model=Prediction)
def predict(movie: Movie):
    """Estima la recaudación de una película en USD.

    Devuelve 503 si todavía no hay un modelo con el alias champion
    (hay que correr los DAGs etl_tmdb_movies y retrain_revenue_model).
    """
    if not model.refresh():
        raise HTTPException(
            status_code=503,
            detail=f"El modelo {model.MODEL_NAME}@{model.MODEL_ALIAS} no está disponible",
        )
    return Prediction(
        revenue=model.predict_revenue(movie),
        model_name=model.MODEL_NAME,
        model_version=model.model_state["version"],
    )

"""
Carga del modelo de recaudación desde MLflow y predicciones para la API.

La API sirve la versión del Model Registry que tiene el alias MODEL_ALIAS (por defecto
"champion"). Cuando el DAG retrain_revenue_model promueve una versión nueva, la API la
detecta y la carga sola: cada MODEL_CHECK_SECONDS consulta a qué versión apunta el alias.

Configuración por variables de entorno:

- MLFLOW_TRACKING_URI: URL del servidor de MLflow (por defecto http://mlflow:5000)
- MODEL_NAME: nombre del modelo registrado (por defecto revenue_regressor)
- MODEL_ALIAS: alias a servir (por defecto champion)
- MODEL_CHECK_SECONDS: cada cuántos segundos revisar el alias (por defecto 30)
"""

import logging
import os
import threading
import time
import pandas as pd
from schema import Movie

logger = logging.getLogger("uvicorn.error")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "revenue_regressor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_CHECK_SECONDS = float(os.getenv("MODEL_CHECK_SECONDS", "30"))

# Estado del modelo, lo completa refresh():
# - model: el pipeline de sklearn entrenado (TransformedTargetRegressor)
# - version: versión del Model Registry que está cargada
# - columns: columnas de entrada del modelo, en orden
# - idiomas: idiomas que conoce el OneHotEncoder del modelo
# - checked_at: cuándo se consultó el alias por última vez (time.monotonic)

model_state = {"model": None, "version": None, "columns": [], "idiomas": [], "checked_at": None}

# Los endpoints corren en varios hilos: el lock evita que dos requests recarguen a la vez
_lock = threading.Lock()

def _champion_version() -> str:
    """Consulta en MLflow a qué versión apunta el alias. Falla si el alias no existe."""

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    return str(client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version)


def _load_version(version: str):
    """Descarga y carga una versión concreta del modelo."""

    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def _set_model(model, version: str) -> None:
    """Guarda el modelo en model_state junto con las columnas e idiomas que espera."""

    # Idiomas vistos en el entrenamiento; cualquier otro se trata como "otro"
    encoder = model.regressor_.named_steps["prep"].named_transformers_["categoricas"]
    model_state.update(
        model=model,
        version=version,
        columns=list(model.feature_names_in_),
        idiomas=list(encoder.categories_[0]),
    )


def refresh(force: bool = False) -> bool:
    """Carga el modelo del alias si cambió de versión o si todavía no hay ninguno.

    Para no consultar MLflow en cada request, solo revisa el alias cada MODEL_CHECK_SECONDS,
    salvo con force=True. Por eso un champion nuevo puede tardar hasta ese tiempo en
    servirse. Si MLflow no responde, se sigue usando el modelo que esté cargado.

    Returns:
        True si hay un modelo cargado al terminar.
    """

    checked_at = model_state["checked_at"]
    if not force and checked_at is not None \
            and time.monotonic() - checked_at < MODEL_CHECK_SECONDS:
        return is_loaded()

    with _lock:
        model_state["checked_at"] = time.monotonic()
        try:
            version = _champion_version()
            if version != model_state["version"]:
                _set_model(_load_version(version), version)
                logger.info("Modelo %s@%s cargado: versión %s", MODEL_NAME, MODEL_ALIAS, version)
        except Exception as e:
            logger.warning("No se pudo actualizar %s@%s: %s", MODEL_NAME, MODEL_ALIAS, e)
    return is_loaded()


def is_loaded() -> bool:
    """True si hay un modelo cargado."""

    return model_state["model"] is not None


def build_features(movie: Movie) -> pd.DataFrame:
    """Arma la entrada del modelo para una película.

    Tiene que dar lo mismo que make_features() en airflow/dags/utils/data.py; lo verifica
    tests/test_api.py. Los géneros y los idiomas conocidos salen del modelo cargado, así
    que siguen los cambios de cada reentrenamiento.

    Args:
        movie: película recibida en el request.

    Returns:
        DataFrame de una fila con las columnas del modelo, en orden.
    """

    genres = {"gen_" + g.lower().replace(" ", "_") for g in movie.genres}
    idioma = movie.original_language.lower()
    row = {
        "budget": movie.budget,
        "popularity": movie.popularity,
        "vote_count": movie.vote_count,
        "n_companias": movie.n_production_companies,
        "runtime": movie.runtime,
        "vote_average": movie.vote_average,
        "anio": movie.release_year,
        "idioma": idioma if idioma in model_state["idiomas"] else "otro",
    }
    for column in model_state["columns"]:
        if column.startswith("gen_"):
            row[column] = int(column in genres)
    return pd.DataFrame([row], columns=model_state["columns"])


def predict_revenue(movie: Movie) -> float:
    """Estima la recaudación de una película. Requiere un modelo cargado (ver is_loaded()).

    Returns:
        Recaudación estimada en USD. El pipeline deshace el log del target.
    """
    
    return float(model_state["model"].predict(build_features(movie))[0])

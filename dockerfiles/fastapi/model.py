"""
Loads the revenue model from MLflow and runs predictions for the FastAPI endpoints.

The model is fetched from the MLflow Model Registry as MODEL_NAME@MODEL_ALIAS; its files are
downloaded from MinIO. Configuration comes from environment variables:

- MLFLOW_TRACKING_URI: MLflow server URL (default: http://mlflow:5000)
- MODEL_NAME: registered model name (default: revenue_regressor)
- MODEL_ALIAS: registered model alias (default: champion)
"""

import logging
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from schema import Movie

logger = logging.getLogger("uvicorn.error")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "revenue_regressor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# Model state, filled by load_model():
# - model: the fitted sklearn pipeline (TransformedTargetRegressor)
# - version: registered model version the alias pointed to at load time
# - columns: input columns expected by the model, in order (from its signature)
# - idiomas: languages known by the model's OneHotEncoder
model_state = {"model": None, "version": None, "columns": [], "idiomas": []}


def load_model():
    """Loads MODEL_NAME@MODEL_ALIAS from MLflow into model_state.

    Errors are logged instead of raised, so the API can start before the model is registered.

    Returns:
        True if the model was loaded, False if it's not available.
    """
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.sklearn.load_model(uri)
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
        columns = mlflow.models.get_model_info(uri).signature.inputs.input_names()
    except Exception as e:
        logger.warning("Model %s not available: %s", uri, e)
        return False

    # Languages seen in training; anything else is mapped to "otro" like in training
    encoder = model.regressor_.named_steps["prep"].named_transformers_["categoricas"]
    model_state.update(model=model, version=version, columns=columns,
                       idiomas=list(encoder.categories_[0]))
    logger.info("Loaded model %s version %s", uri, version)
    return True


def is_loaded() -> bool:
    """Returns True if load_model() has loaded a model."""
    return model_state["model"] is not None


def build_features(movie: Movie) -> pd.DataFrame:
    """Builds the model input for a movie, same as construir_features() in notebooks/train.ipynb.

    Genre flags and known languages come from the loaded model, so they follow retrains.

    Args:
        movie: Movie from the request.

    Returns:
        Single-row DataFrame with the model's input columns, in order.
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
    """Predicts a movie's revenue. Requires a loaded model (see is_loaded()).

    Args:
        movie: Movie from the request.

    Returns:
        Predicted revenue in USD. The pipeline undoes the log transform of the target.
    """
    return float(model_state["model"].predict(build_features(movie))[0])

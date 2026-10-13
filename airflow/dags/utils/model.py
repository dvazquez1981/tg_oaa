"""
Pipeline del modelo de recaudación y manejo de champion/challenger en MLflow.

El "champion" es la versión del modelo en producción (la que usa la API) y el "challenger"
es un modelo recién entrenado que compite contra él. Son alias del Model Registry de MLflow.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from utils import config, s3


SKEWED = ["budget", "popularity", "vote_count", "n_companias"]
NUMERIC = ["runtime", "vote_average", "anio"]
CATEGORICAL = ["idioma"]
DEFAULT_PARAMS = {"random_state": config.RANDOM_STATE, "n_jobs": -1}
SKOPS_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"]


def build_model(params: dict, columns) -> TransformedTargetRegressor:
    """Arma el pipeline completo, sin entrenar.

    Preprocesa las features y entrena un XGBoost sobre log(1 + revenue). Al predecir,
    TransformedTargetRegressor deshace el log, así que las predicciones salen en dólares.
    Los nombres de los pasos ("prep", "modelo", "categoricas") los usa la API para leer
    el modelo cargado, así que no hay que cambiarlos.

    Args:
        params: hiperparámetros de XGBRegressor.
        columns: columnas de entrenamiento, para detectar las gen_*.
    """

    binarias = [c for c in columns if c.startswith("gen_")]
    preprocesador = ColumnTransformer([
        ("sesgadas", Pipeline([("log", FunctionTransformer(np.log1p)),
                               ("escala", StandardScaler())]), SKEWED),
        ("numericas", StandardScaler(), NUMERIC),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
        ("binarias", "passthrough", binarias),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocesador), ("modelo", XGBRegressor(**params))]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def load_split():
    """Lee de S3 los conjuntos que deja el ETL.

    Returns:
        x_train, y_train, x_test, y_test. Los y como Series.
    """

    return (
        s3.read_parquet(config.X_TRAIN_PATH),
        s3.read_parquet(config.Y_TRAIN_PATH)[config.TARGET],
        s3.read_parquet(config.X_TEST_PATH),
        s3.read_parquet(config.Y_TEST_PATH)[config.TARGET],
    )


def mlflow_client():
    """Apunta MLflow al servidor del proyecto y devuelve un cliente."""

    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.MlflowClient()


def load_alias(alias: str):
    """Carga la versión del modelo que tiene el alias, o None si no existe."""

    import mlflow.sklearn
    from mlflow.exceptions import MlflowException

    mlflow_client()
    try:
        return mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{alias}")
    except MlflowException:
        return None


def has_champion() -> bool:
    """True si en el Model Registry hay una versión con el alias champion."""

    from mlflow.exceptions import MlflowException

    try:
        mlflow_client().get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION)
        return True
    except MlflowException:
        return False


def champion_params() -> dict:
    """Hiperparámetros del champion actual, o DEFAULT_PARAMS si todavía no hay champion."""

    champion = load_alias(config.CHAMPION)
    if champion is None:
        print("ℹ️ No hay champion, se usan los hiperparámetros por defecto")
        return dict(DEFAULT_PARAMS)
    print("🏆 Se usan los hiperparámetros del champion")
    return champion.regressor_.named_steps["modelo"].get_params()


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """MAE del modelo sobre el set de testeo, en dólares.

    Si el ETL cambió los géneros frecuentes, el set puede no tener alguna columna gen_*
    con la que se entrenó el modelo: se completa con 0 (película sin ese género).
    """

    x_test = x_test.reindex(columns=model.regressor_.feature_names_in_, fill_value=0)
    return float(mean_absolute_error(y_test, model.predict(x_test)))


def train_and_register(params: dict, origin: str) -> dict:
    """Entrena un modelo nuevo, lo registra en MLflow y le pone el alias challenger.

    Args:
        params: hiperparámetros de XGBRegressor.
        origin: de dónde salieron los hiperparámetros ("champion", "optuna"...). Queda
            como tag de la corrida para saber cómo se entrenó cada versión.

    Returns:
        La versión registrada y el run_id de MLflow.
    """

    import datetime
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    x_train, y_train, x_test, y_test = load_split()
    model = build_model(params, x_train.columns)
    print(f"🏋️ Entrenando con {len(x_train):,} películas...")
    model.fit(x_train, y_train)

    client = mlflow_client()
    mlflow.set_experiment(config.TRAIN_EXPERIMENT)
    run_name = "challenger_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("origen_hiperparametros", origin)
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metric("test_mae", evaluate(model, x_test, y_test))
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(x_train, model.predict(x_train)),
            input_example=x_train.head(5),
            registered_model_name=config.MODEL_NAME,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    version = str(info.registered_model_version)
    client.set_registered_model_alias(config.MODEL_NAME, config.CHALLENGER, version)
    print(f"💾 Challenger registrado: {config.MODEL_NAME} versión {version}")
    return {"version": version, "run_id": info.run_id}


def set_champion(version: str) -> None:
    """Pasa el alias champion a la versión indicada y le saca el alias challenger."""

    client = mlflow_client()
    client.delete_registered_model_alias(config.MODEL_NAME, config.CHALLENGER)
    client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION, version)


def drop_challenger() -> None:
    """Le saca el alias challenger al modelo, sin tocar al champion."""

    mlflow_client().delete_registered_model_alias(config.MODEL_NAME, config.CHALLENGER)

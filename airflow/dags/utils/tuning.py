"""
Búsqueda de hiperparámetros del modelo con Optuna, registrada en MLflow.

Usa el mismo espacio de búsqueda y la misma validación cruzada que notebooks/01_TRAIN.ipynb.
Cada combinación probada queda como un run hijo dentro de un run padre en el experimento
config.TUNING_EXPERIMENT, así se pueden comparar todas en la interfaz de MLflow.
"""
import datetime

import optuna
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from utils import config
from utils import model as model_utils


def suggest_params(trial: optuna.Trial) -> dict:
    """Hiperparámetros de XGBRegressor que propone Optuna en cada intento."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }


def cv_mae(params: dict, x: pd.DataFrame, y: pd.Series, folds: int) -> float:
    """MAE promedio por validación cruzada del pipeline completo, en dólares.

    Cada modelo usa un solo hilo y los folds se entrenan en paralelo.
    """
    regresor = model_utils.build_model(
        {**params, "random_state": config.RANDOM_STATE, "n_jobs": 1}, x.columns
    )
    cv = KFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(regresor, x, y, cv=cv,
                             scoring="neg_mean_absolute_error", n_jobs=-1)
    return float(-scores.mean())


def search(x: pd.DataFrame, y: pd.Series, n_trials: int, folds: int = 5, on_trial=None):
    """Busca los hiperparámetros que minimizan el MAE de validación cruzada.

    Args:
        x, y: conjunto de entrenamiento. El de testeo no se usa, para no sesgar la
            comparación posterior contra el champion.
        n_trials: cantidad de combinaciones a probar.
        folds: particiones de la validación cruzada.
        on_trial: función opcional on_trial(numero, params, mae) que se llama después
            de cada intento; la usa search_and_log para registrar en MLflow.

    Returns:
        (mejores hiperparámetros, su MAE de validación cruzada). Los hiperparámetros
        incluyen random_state y n_jobs, listos para model.build_model.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )

    def objetivo(trial):
        params = suggest_params(trial)
        mae = cv_mae(params, x, y, folds)
        print(f"   intento {trial.number + 1}/{n_trials}: MAE CV = {mae:,.0f}")
        if on_trial is not None:
            on_trial(trial.number, params, mae)
        return mae

    study.optimize(objetivo, n_trials=n_trials)
    return {**study.best_params, **model_utils.DEFAULT_PARAMS}, float(study.best_value)


def search_and_log(n_trials: int, folds: int = 5) -> dict:
    """Corre la búsqueda sobre el train del ETL y la registra en MLflow.

    Returns:
        Los mejores hiperparámetros, para entrenar el challenger con ellos.
    """
    import mlflow

    x_train, y_train, _, _ = model_utils.load_split()
    model_utils.mlflow_client()
    mlflow.set_experiment(config.TUNING_EXPERIMENT)

    run_name = "optuna_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"🔎 Buscando hiperparámetros: {n_trials} intentos, CV de {folds} folds, "
          f"{len(x_train):,} películas")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"n_trials": n_trials, "cv_folds": folds,
                           "train_rows": len(x_train)})

        def registrar_intento(numero, params, mae):
            with mlflow.start_run(run_name=f"intento_{numero:03d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_mae", mae)

        best_params, best_mae = search(x_train, y_train, n_trials, folds, registrar_intento)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_mae", best_mae)

    print(f"🏁 Mejor MAE CV: {best_mae:,.0f} con {best_params}")
    return best_params

"""Tests de utils/tuning.py: la búsqueda con Optuna, sin S3 ni MLflow."""
import numpy as np
import pandas as pd
import pytest

from utils import config, model, tuning


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    n = 150
    budget = rng.lognormal(16, 1, n)
    x = pd.DataFrame({
        "budget": budget,
        "popularity": rng.lognormal(2, 1, n),
        "vote_count": rng.integers(1, 5000, n),
        "n_companias": rng.integers(0, 5, n),
        "runtime": rng.integers(80, 180, n),
        "vote_average": rng.uniform(1, 10, n),
        "anio": rng.integers(1980, 2024, n),
        "idioma": rng.choice(["en", "fr", "otro"], n),
        "gen_drama": rng.integers(0, 2, n),
    })
    return x, pd.Series(budget * rng.uniform(0.5, 4, n))


def test_search_registra_cada_intento_y_devuelve_el_mejor(train):
    x, y = train
    intentos = []
    best_params, best_mae = tuning.search(
        x, y, n_trials=3, folds=2, on_trial=lambda n, p, mae: intentos.append(mae)
    )

    assert len(intentos) == 3
    assert best_mae == pytest.approx(min(intentos))
    # Incluye los hiperparámetros buscados y los fijos
    assert {"n_estimators", "learning_rate", "max_depth"} <= set(best_params)
    assert best_params["random_state"] == config.RANDOM_STATE


def test_los_mejores_hiperparametros_sirven_para_entrenar(train):
    x, y = train
    best_params, _ = tuning.search(x, y, n_trials=2, folds=2)
    regresor = model.build_model(best_params, x.columns).fit(x, y)
    assert regresor.predict(x).shape == (len(x),)


def test_search_es_reproducible(train):
    x, y = train
    assert tuning.search(x, y, n_trials=2, folds=2) == tuning.search(x, y, n_trials=2, folds=2)

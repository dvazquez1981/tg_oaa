"""Tests de utils/model.py: el pipeline entrena y predice, sin S3 ni MLflow."""
import numpy as np
import pandas as pd
import pytest
from utils import config, model

@pytest.fixture
def features():
    rng = np.random.default_rng(0)
    n = 200
    budget = rng.lognormal(16, 1, n)
    return pd.DataFrame({
        "budget": budget,
        "popularity": rng.lognormal(2, 1, n),
        "vote_count": rng.integers(1, 5000, n),
        "n_companias": rng.integers(0, 5, n),
        "runtime": rng.integers(80, 180, n),
        "vote_average": rng.uniform(1, 10, n),
        "anio": rng.integers(1980, 2024, n),
        "idioma": rng.choice(["en", "fr", "otro"], n),
        "gen_drama": rng.integers(0, 2, n),
        "gen_comedy": rng.integers(0, 2, n),
        config.TARGET: budget * rng.uniform(0.5, 4, n),
    })


def test_build_model_entrena_y_predice_en_dolares(features):
    x, y = features.drop(columns=config.TARGET), features[config.TARGET]
    regresor = model.build_model({"n_estimators": 20, **model.DEFAULT_PARAMS}, x.columns)
    regresor.fit(x, y)

    predicciones = regresor.predict(x)
    assert predicciones.shape == (len(x),)
    # El pipeline deshace el log: las predicciones están en la escala de revenue
    assert predicciones.min() > 0
    assert np.median(predicciones) == pytest.approx(y.median(), rel=0.5)


def test_nombres_de_pasos_que_usa_la_api(features):
    x, y = features.drop(columns=config.TARGET), features[config.TARGET]
    regresor = model.build_model({"n_estimators": 5}, x.columns).fit(x, y)

    pasos = regresor.regressor_.named_steps
    assert set(pasos) == {"prep", "modelo"}
    assert "categoricas" in pasos["prep"].named_transformers_


def test_evaluate_completa_generos_faltantes(features):
    x, y = features.drop(columns=config.TARGET), features[config.TARGET]
    regresor = model.build_model({"n_estimators": 5}, x.columns).fit(x, y)

    # Si el ETL dejó de generar un género, evaluate igual funciona
    mae = model.evaluate(regresor, x.drop(columns="gen_comedy"), y)
    assert mae > 0

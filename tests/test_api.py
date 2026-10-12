"""Tests de la API (dockerfiles/fastapi) sin MLflow: el modelo se entrena acá y se inyecta."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import app as api
import model as api_model
from schema import Movie
from utils import config, data
from utils import model as train_model

GENEROS = ["Drama", "Comedy", "Action", "Science Fiction", "Horror", "Crime"]
IDIOMAS = ["en", "fr", "es", "ja"]
EJEMPLO = {
    "budget": 25_000_000, "popularity": 16.3, "vote_count": 3427, "vote_average": 7.2,
    "runtime": 143, "release_year": 2006, "original_language": "en",
    "genres": ["Drama", "Crime"], "n_production_companies": 6,
}


@pytest.fixture(scope="module")
def peliculas():
    """Dataset limpio de 300 películas sintéticas, con el formato de TMDB."""
    rng = np.random.default_rng(0)
    n = 300
    budget = rng.integers(100_000, 200_000_000, n)
    crudo = pd.DataFrame({
        "id": range(n), "title": "x", "status": "Released",
        "release_date": pd.to_datetime(rng.integers(0, 15000, n), unit="D",
                                       origin="1980-01-01").strftime("%Y-%m-%d"),
        "revenue": (budget * rng.uniform(0.3, 5, n)).round(),
        "budget": budget, "runtime": rng.integers(70, 200, n),
        "popularity": rng.uniform(0.5, 80, n), "vote_average": rng.uniform(1, 10, n).round(1),
        "vote_count": rng.integers(1, 9000, n),
        "original_language": rng.choice(IDIOMAS, n),
        "genres": [", ".join(rng.choice(GENEROS, rng.integers(1, 4), replace=False)) for _ in range(n)],
        "production_companies": [", ".join(f"C{j}" for j in range(k)) or None
                                 for k in rng.integers(0, 4, n)],
    })
    return data.clean(crudo)


@pytest.fixture(scope="module")
def modelo(peliculas):
    """Modelo entrenado con el mismo pipeline que el DAG de reentrenamiento."""
    features = data.make_features(peliculas)
    x, y = features.drop(columns=config.TARGET), features[config.TARGET]
    return train_model.build_model({"n_estimators": 30, **train_model.DEFAULT_PARAMS}, x.columns).fit(x, y)


@pytest.fixture
def registro(monkeypatch, modelo):
    """Simula el Model Registry: un dict {versión: modelo} y el alias champion."""
    estado = {"champion": "1", "versiones": {"1": modelo}, "caido": False}

    def champion_version():
        if estado["caido"]:
            raise ConnectionError("MLflow no responde")
        if estado["champion"] is None:
            raise LookupError("no existe el alias")
        return estado["champion"]

    monkeypatch.setattr(api_model, "_champion_version", champion_version)
    monkeypatch.setattr(api_model, "_load_version", lambda v: estado["versiones"][v])
    monkeypatch.setattr(api_model, "MODEL_CHECK_SECONDS", 0)
    monkeypatch.setattr(api_model, "model_state", {
        "model": None, "version": None, "columns": [], "idiomas": [], "checked_at": None,
    })
    return estado


@pytest.fixture
def cliente(registro):
    with TestClient(api.app) as c:
        yield c


def test_predict(cliente):
    r = cliente.post("/predict", json=EJEMPLO)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["revenue"] > 0
    assert (cuerpo["model_name"], cuerpo["model_version"]) == ("revenue_regressor", "1")


def test_predict_idioma_y_genero_desconocidos(cliente):
    r = cliente.post("/predict", json=EJEMPLO | {"original_language": "xx", "genres": ["Inventado"]})
    assert r.status_code == 200


@pytest.mark.parametrize("cambio", [{"budget": -1}, {"vote_average": 11}, {"runtime": 0}])
def test_predict_valida_la_entrada(cliente, cambio):
    assert cliente.post("/predict", json=EJEMPLO | cambio).status_code == 422


def test_sin_modelo_devuelve_503(registro):
    registro["champion"] = None
    with TestClient(api.app) as c:
        assert c.post("/predict", json=EJEMPLO).status_code == 503
        assert c.get("/health").json()["status"] == "sin_modelo"


def test_carga_el_champion_nuevo_sin_reiniciar(cliente, registro, modelo):
    assert cliente.get("/health").json()["model_version"] == "1"

    # El DAG promueve la versión 2
    registro["versiones"]["2"] = modelo
    registro["champion"] = "2"

    assert cliente.post("/predict", json=EJEMPLO).json()["model_version"] == "2"
    assert cliente.get("/health").json() == {
        "status": "ok", "model_name": "revenue_regressor",
        "model_alias": "champion", "model_version": "2",
    }


def test_si_mlflow_se_cae_sigue_sirviendo(cliente, registro):
    registro["caido"] = True
    r = cliente.post("/predict", json=EJEMPLO)
    assert r.status_code == 200
    assert r.json()["model_version"] == "1"


def test_features_iguales_a_las_del_entrenamiento(peliculas, modelo):
    """build_features de la API tiene que dar lo mismo que make_features del ETL."""
    api_model._set_model(modelo, "1")
    esperado = data.make_features(peliculas).drop(columns=config.TARGET)

    for i, fila in peliculas.head(50).iterrows():
        pelicula = Movie(
            budget=int(fila["budget"]), popularity=fila["popularity"],
            vote_count=int(fila["vote_count"]), vote_average=fila["vote_average"],
            runtime=int(fila["runtime"]), release_year=fila["release_date"].year,
            original_language=fila["original_language"],
            genres=data.parse_list(fila["genres"]),
            n_production_companies=len(data.parse_list(fila["production_companies"])),
        )
        obtenido = api_model.build_features(pelicula).iloc[0]
        pd.testing.assert_series_equal(
            obtenido, esperado.iloc[i][obtenido.index], check_names=False, check_dtype=False,
        )

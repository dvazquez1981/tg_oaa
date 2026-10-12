"""Tests de utils/data.py: limpieza, features y split, sin S3 ni MLflow."""
import pandas as pd
import pytest
from utils import config, data

def pelicula(**cambios):
    """Una película válida del dataset crudo; los argumentos pisan sus valores."""
    base = {
        "id": 1, "title": "Película", "status": "Released", "release_date": "2010-05-01",
        "revenue": 1_000_000, "budget": 200_000, "runtime": 120, "popularity": 10.0,
        "vote_average": 7.0, "vote_count": 500, "original_language": "en",
        "genres": "Drama, Crime", "production_companies": "A, B",
    }
    return {**base, **cambios}


@pytest.fixture
def crudo():
    return pd.DataFrame([
        pelicula(id=1),
        pelicula(id=2, status="Rumored"),
        pelicula(id=3, revenue=0),
        pelicula(id=4, budget=0),
        pelicula(id=5, release_date="fecha rota"),
        pelicula(id=6, genres=None),
        pelicula(id=7, vote_count=None),
        pelicula(id=1),  # duplicada
        pelicula(id=8, original_language="fr", genres="Comedy",
                 production_companies=None, release_date="1999-12-31"),
    ])


def test_parse_list():
    assert data.parse_list("Drama, Crime") == ["Drama", "Crime"]
    assert data.parse_list("Drama,, ") == ["Drama"]
    assert data.parse_list(None) == []


def test_genre_column():
    assert data.genre_column("Science Fiction") == "gen_science_fiction"


def test_clean_filtra_y_saca_duplicados(crudo):
    limpio = data.clean(crudo)
    assert sorted(limpio["id"]) == [1, 8]
    assert pd.api.types.is_datetime64_any_dtype(limpio["release_date"])


def test_clean_no_modifica_la_entrada(crudo):
    original = crudo.copy()
    data.clean(crudo)
    pd.testing.assert_frame_equal(crudo, original)


def test_make_features(crudo):
    features = data.make_features(data.clean(crudo))

    assert len(features) == 2
    assert features.notna().all().all()
    assert {"budget", "anio", "idioma", "n_companias", config.TARGET} <= set(features.columns)
    assert {"gen_drama", "gen_crime", "gen_comedy"} <= set(features.columns)

    drama, comedia = features.iloc[0], features.iloc[1]
    assert (drama["gen_drama"], drama["gen_comedy"]) == (1, 0)
    assert (comedia["gen_drama"], comedia["gen_comedy"]) == (0, 1)
    assert comedia["n_companias"] == 0
    assert comedia["anio"] == 1999


def test_make_features_agrupa_idiomas_raros(monkeypatch, crudo):
    monkeypatch.setattr(config, "N_TOP_LANGUAGES", 1)
    features = data.make_features(data.clean(crudo))
    # Queda solo el idioma más frecuente y el otro pasa a "otro"
    assert "otro" in set(features["idioma"])
    assert features["idioma"].nunique() == 2


def test_split():
    df = pd.DataFrame({"budget": range(100), "gen_drama": [0, 1] * 50, config.TARGET: range(100)})
    x_train, x_test, y_train, y_test = data.split(df, test_size=0.2)

    assert (len(x_train), len(x_test)) == (80, 20)
    assert config.TARGET not in x_train.columns
    assert list(y_train.columns) == [config.TARGET]
    # random_state fijo: la división es reproducible
    pd.testing.assert_frame_equal(x_test, data.split(df, test_size=0.2)[1])

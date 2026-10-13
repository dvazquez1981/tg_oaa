"""
Descarga, limpieza, features y split del dataset de películas de TMDB.

Las funciones de limpieza y features reciben y devuelven DataFrames, sin leer ni escribir
en S3, así se pueden probar con pytest sin levantar los servicios.
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import config

# Columnas del CSV de Kaggle que usamos
COLUMNS = [
    "id", "title", "status", "release_date", "revenue", "budget",
    "runtime", "popularity", "vote_average", "vote_count",
    "original_language", "genres", "production_companies",
]


def download_tmdb() -> pd.DataFrame:
    """Descarga el dataset de TMDB de Kaggle y lo carga con las columnas que usamos.

    El CSV pesa unos 600 MB, así que se borra del disco después de leerlo.

    Returns:
        El dataset crudo, una fila por película.
    """

    cache = "/tmp/kagglehub"
    # kagglehub lee esta variable al importarse
    os.environ["KAGGLEHUB_CACHE"] = cache
    import kagglehub

    print(f"📥 Descargando {config.KAGGLE_DATASET} de Kaggle...")
    try:
        csv = kagglehub.dataset_download(config.KAGGLE_DATASET, path=config.KAGGLE_FILE)
        return pd.read_csv(csv, usecols=COLUMNS)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Se queda con las películas estrenadas que tienen datos financieros válidos.

    Imprime cuántas películas quedan después de cada filtro, para verlo en el log.

    Args:
        df: dataset crudo, con las columnas de COLUMNS.

    Returns:
        Dataset filtrado, sin películas duplicadas y con release_date como fecha.
    """

    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    filtros = {
        "status == 'Released'": df["status"].eq("Released"),
        "revenue > 0": df["revenue"] > 0,
        "budget > 0": df["budget"] > 0,
        "runtime > 0": df["runtime"] > 0,
        "vote_count > 0": df["vote_count"] > 0,
        "vote_average > 0": df["vote_average"] > 0,
        "fecha no nula": df["release_date"].notna(),
        "género no nulo": df["genres"].notna(),
    }

    print(f"🧹 Dataset completo: {len(df):,}")
    mascara = pd.Series(True, index=df.index)
    for nombre, condicion in filtros.items():
        # Un valor nulo cuenta como que no cumple el filtro
        mascara &= condicion.fillna(False)
        print(f"   {nombre}: {mascara.sum():,}")

    return df[mascara].drop_duplicates(subset="id").reset_index(drop=True)


def parse_list(valor) -> list[str]:
    """Convierte un texto separado por comas en lista: 'Drama, Crime' -> ['Drama', 'Crime']."""

    if pd.isna(valor):
        return []
    return [x.strip() for x in str(valor).split(",") if x.strip()]


def genre_column(genero: str) -> str:
    """Nombre de la columna indicadora de un género: 'Science Fiction' -> 'gen_science_fiction'."""

    return "gen_" + genero.lower().replace(" ", "_")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Arma la matriz de features a partir del dataset limpio.

    Los géneros y los idiomas más frecuentes se calculan con los mismos datos, así que
    las columnas gen_* pueden cambiar si cambia el dataset.

    Args:
        df: dataset limpio (salida de clean).

    Returns:
        Una fila por película: features numéricas, idioma, una columna 0/1 por género
        frecuente y el target (revenue). Sin valores nulos.
    """

    generos = df["genres"].apply(parse_list)
    top_generos = generos.explode().value_counts().head(config.N_TOP_GENRES).index.tolist()
    top_idiomas = df["original_language"].value_counts().head(config.N_TOP_LANGUAGES).index

    features = pd.DataFrame(index=df.index)

    # Sesgadas: el pipeline del modelo les aplica log
    features["budget"] = df["budget"]
    features["popularity"] = df["popularity"]
    features["vote_count"] = df["vote_count"]
    features["n_companias"] = df["production_companies"].apply(parse_list).str.len()

    # Numéricas
    features["runtime"] = df["runtime"]
    features["vote_average"] = df["vote_average"]
    features["anio"] = df["release_date"].dt.year

    # Categórica: los idiomas poco frecuentes van a "otro"
    features["idioma"] = df["original_language"].where(
        df["original_language"].isin(top_idiomas), "otro"
    )

    # Un indicador 0/1 por género frecuente
    for genero in top_generos:
        features[genre_column(genero)] = generos.apply(lambda gs, g=genero: int(g in gs))

    features[config.TARGET] = df[config.TARGET]
    return features.dropna().reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float):
    """Separa la matriz de features en entrenamiento y testeo.

    Args:
        df: salida de make_features.
        test_size: proporción del dataset que va a testeo, entre 0 y 1.

    Returns:
        x_train, x_test, y_train, y_test. Los y son DataFrames de una columna, para
        poder guardarlos en parquet.
    """

    x = df.drop(columns=config.TARGET)
    y = df[[config.TARGET]]
    return train_test_split(x, y, test_size=test_size, random_state=config.RANDOM_STATE)


def log_etl_run(x_train, x_test, y_train, test_size: float) -> None:
    """Registra la corrida del ETL en MLflow: tamaños, columnas y el dataset de entrenamiento.

    El dataset se registra como referencia (esquema, origen y hash), no se copian los datos.
    """

    import datetime
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.ETL_EXPERIMENT)
    run_name = "etl_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "observations": len(x_train) + len(x_test),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "test_size": test_size,
            "features": x_train.columns.tolist(),
        })
        dataset = mlflow.data.from_pandas(
            x_train.join(y_train), source=config.X_TRAIN_PATH,
            targets=config.TARGET, name="tmdb_train",
        )
        mlflow.log_input(dataset, context="training")

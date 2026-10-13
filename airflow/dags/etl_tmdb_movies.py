"""
ETL del dataset de películas de TMDB.

Descarga el dataset de Kaggle, lo limpia, arma las features y lo separa en train/test,
guardando cada etapa en el bucket "data". Al terminar actualiza el asset TMDB_DATA, que
dispara el DAG retrain_revenue_model.

Corre solo una vez, apenas se levanta Airflow por primera vez (schedule "@once"), así el
sistema arranca con datos y modelo sin intervención. Después se puede ejecutar a mano
cuando se quiera actualizar el dataset.

Acá solo se define el flujo; la lógica está en utils/data.py.
"""

import datetime
from airflow.sdk import Asset, dag, task
from common import ML_VENV
from utils import config

TMDB_DATA = Asset(config.TMDB_ASSET_URI)

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="etl_tmdb_movies",
    description="ETL del dataset de películas de TMDB",
    default_args=default_args,
    schedule="@once",                               # una vez, al primer arranque
    start_date=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    catchup=False,
    is_paused_upon_creation=False,                  # activo sin tener que encenderlo
    max_active_runs=1,
    dagrun_timeout=datetime.timedelta(hours=1),
    tags=["ETL", "TMDB"],
    params={"test_size": 0.2},
)
def etl_tmdb_movies():

    @task.external_python(task_id="obtain_original_data", **ML_VENV)
    def get_data():
        """Descarga el dataset de Kaggle y lo guarda crudo en S3."""

        from utils import config, data, s3

        s3.ensure_buckets(*config.BUCKETS)
        return s3.write_parquet(data.download_tmdb(), config.RAW_PATH)

    @task.external_python(task_id="clean_data", **ML_VENV)
    def clean_data(path):
        """Se queda con las películas estrenadas que tienen datos financieros."""

        from utils import config, data, s3

        return s3.write_parquet(data.clean(s3.read_parquet(path)), config.CLEAN_PATH)

    @task.external_python(task_id="make_features", multiple_outputs=True, **ML_VENV)
    def make_features(path):
        """Arma la matriz de features y devuelve su ruta y su forma."""

        from utils import config, data, s3

        features = data.make_features(s3.read_parquet(path))
        return {
            "path": s3.write_parquet(features, config.FEATURES_PATH),
            "observations": features.shape[0],
            "columns": features.shape[1],
        }

    @task.external_python(
        task_id="split_dataset", multiple_outputs=True, outlets=[TMDB_DATA], **ML_VENV
    )
    def split_dataset(path, obs, col, test_size):
        """Separa en train/test, guarda los cuatro archivos y registra la corrida en MLflow."""
        
        from utils import config, data, s3

        df = s3.read_parquet(path)
        # Control: el archivo tiene que ser el que generó la tarea anterior
        assert df.shape == (obs, col), "⚠️ La forma del dataset no coincide con lo esperado."

        # Llega como texto porque viene de una plantilla de Jinja
        test_size = float(test_size)
        x_train, x_test, y_train, y_test = data.split(df, test_size)

        paths = {
            "x_train": s3.write_parquet(x_train, config.X_TRAIN_PATH),
            "y_train": s3.write_parquet(y_train, config.Y_TRAIN_PATH),
            "x_test": s3.write_parquet(x_test, config.X_TEST_PATH),
            "y_test": s3.write_parquet(y_test, config.Y_TEST_PATH),
        }
        data.log_etl_run(x_train, x_test, y_train, test_size)

        print(f"✅ Train: {len(x_train):,} películas, test: {len(x_test):,} películas")
        return paths

    # 🧩 Encadenamiento: cada tarea recibe la ruta que devuelve la anterior
    raw_path = get_data()
    clean_path = clean_data(raw_path)
    features = make_features(clean_path)
    split_dataset(
        features["path"], features["observations"], features["columns"],
        "{{ params.test_size }}",
    )


etl_tmdb_movies()

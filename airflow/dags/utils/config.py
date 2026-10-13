"""
Rutas, nombres y constantes compartidos por los DAGs.

Solo usa la librería estándar: lo importan tanto los archivos de los DAGs (Python de
Airflow) como las tareas (virtualenv /opt/ml-venv).
"""
import os

# Python del virtualenv de ML, creado en dockerfiles/airflow/Dockerfile
ML_PYTHON = "/opt/ml-venv/bin/python"

# MLflow: la URL viene del docker-compose; el valor por defecto es el del servicio
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ETL_EXPERIMENT = "tmdb_revenue_etl"
TRAIN_EXPERIMENT = "tmdb_revenue_regressor"
TUNING_EXPERIMENT = "tmdb_revenue_tuning"
MODEL_NAME = "revenue_regressor"
CHAMPION = "champion"
CHALLENGER = "challenger"

# Búsqueda de hiperparámetros (valores por defecto; el DAG los deja cambiar al ejecutarlo)
TUNING_TRIALS = 50
TUNING_FOLDS = 5

# Fuente de datos
KAGGLE_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
KAGGLE_FILE = "TMDB_movie_dataset_v11.csv"

# Buckets que usa el proyecto: datos y artefactos de MLflow
BUCKETS = ("data", "mlflow")

# Rutas en el bucket "data", una por etapa del ETL
RAW_PATH = "s3://data/raw/tmdb_movies.parquet"
CLEAN_PATH = "s3://data/processed/tmdb_movies_clean.parquet"
FEATURES_PATH = "s3://data/processed/tmdb_features.parquet"
X_TRAIN_PATH = "s3://data/final/train/tmdb_X_train.parquet"
Y_TRAIN_PATH = "s3://data/final/train/tmdb_y_train.parquet"
X_TEST_PATH = "s3://data/final/test/tmdb_X_test.parquet"
Y_TEST_PATH = "s3://data/final/test/tmdb_y_test.parquet"

# Asset que actualiza el ETL y dispara el reentrenamiento. Es solo un identificador:
# Airflow no revisa el bucket.
TMDB_ASSET_URI = "s3://data/final/tmdb"

# Features
TARGET = "revenue"
N_TOP_GENRES = 12
N_TOP_LANGUAGES = 8
RANDOM_STATE = 42

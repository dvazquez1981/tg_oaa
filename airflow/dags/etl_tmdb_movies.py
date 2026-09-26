import datetime

from airflow.sdk import Asset, dag, task

# Virtualenv con las mismas versiones que la API (ver dockerfiles/airflow/Dockerfile)
ML_PYTHON = "/opt/ml-venv/bin/python"

# Cuando este DAG actualiza el asset, se dispara retrain_revenue_model
TMDB_DATA = Asset("s3://data/final/tmdb")

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="etl_tmdb_movies",
    description="ETL del dataset de películas de TMDB",
    default_args=default_args,
    schedule=None,
    catchup=False,
    dagrun_timeout=datetime.timedelta(hours=1),
    tags=["ETL", "TMDB"],
    params={"test_size": 0.2},
)
def etl_tmdb_movies():

    @task.external_python(
        task_id="obtain_original_data", python=ML_PYTHON, expect_airflow=False
    )
    def get_data():
        """
        Descarga el dataset de Kaggle y lo guarda en S3
        """
        import io  # noqa: PLC0415
        import os  # noqa: PLC0415
        import shutil  # noqa: PLC0415

        os.environ["KAGGLEHUB_CACHE"] = "/tmp/kagglehub"

        import boto3  # noqa: PLC0415
        import kagglehub  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        columnas = [
            "id", "title", "status", "release_date", "revenue", "budget",
            "runtime", "popularity", "vote_average", "vote_count",
            "original_language", "genres", "production_companies",
        ]

        print("📥 Descargando dataset de Kaggle...")
        try:
            csv = kagglehub.dataset_download(
                "asaniczka/tmdb-movies-dataset-2023-930k-movies",
                path="TMDB_movie_dataset_v11.csv",
            )
            df = pd.read_csv(csv, usecols=columnas)
        finally:
            # El CSV pesa ~600 MB, no lo dejamos en el contenedor
            shutil.rmtree(os.environ["KAGGLEHUB_CACHE"], ignore_errors=True)

        key = "raw/tmdb_movies.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        boto3.client("s3").put_object(Bucket="data", Key=key, Body=buf.getvalue())

        print(f"💾 {len(df):,} películas guardadas en: s3://data/{key}")
        return f"s3://data/{key}"

    @task.external_python(
        task_id="clean_data", python=ML_PYTHON, expect_airflow=False
    )
    def clean_data(s3_path):
        """
        Se queda con las películas estrenadas que tienen datos financieros
        """
        import io  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        s3 = boto3.client("s3")

        print(f"📂 Cargando dataset desde: {s3_path}")
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        filtros = [
            ("status == 'Released'", df["status"].eq("Released")),
            ("revenue > 0", df["revenue"] > 0),
            ("budget > 0", df["budget"] > 0),
            ("runtime > 0", df["runtime"] > 0),
            ("vote_count > 0", df["vote_count"] > 0),
            ("vote_average > 0", df["vote_average"] > 0),
            ("fecha no nula", df["release_date"].notna()),
            ("género no nulo", df["genres"].notna()),
        ]

        print("🧹 Aplicando filtros...")
        print(f"   Dataset completo: {len(df):,}")
        mascara = pd.Series(True, index=df.index)
        for nombre, condicion in filtros:
            mascara &= condicion.fillna(False)
            print(f"   {nombre}: {mascara.sum():,}")

        df = df[mascara].drop_duplicates(subset="id").reset_index(drop=True)

        out_key = "processed/tmdb_movies_clean.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        s3.put_object(Bucket="data", Key=out_key, Body=buf.getvalue())

        print(f"💾 Dataset limpio guardado en: s3://data/{out_key}")
        return f"s3://data/{out_key}"

    @task.external_python(
        task_id="make_features",
        python=ML_PYTHON,
        expect_airflow=False,
        multiple_outputs=True,
    )
    def make_features(s3_path):
        """
        Arma la matriz de features, igual que en notebooks/train.ipynb
        """
        import io  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        s3 = boto3.client("s3")

        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

        def parsear_lista(valor):
            """Convierte 'Drama, Crimen' en ['Drama', 'Crimen']"""
            if pd.isna(valor):
                return []
            return [x.strip() for x in str(valor).split(",") if x.strip()]

        generos = df["genres"].apply(parsear_lista)
        top_generos = generos.explode().value_counts().head(12).index.tolist()
        top_idiomas = df["original_language"].value_counts().head(8).index.tolist()

        print("🏷️ Generando features...")
        features = pd.DataFrame(index=df.index)

        # Sesgadas (el pipeline les aplica log)
        features["budget"] = df["budget"]
        features["popularity"] = df["popularity"]
        features["vote_count"] = df["vote_count"]
        features["n_companias"] = df["production_companies"].apply(parsear_lista).str.len()

        # Numéricas
        features["runtime"] = df["runtime"]
        features["vote_average"] = df["vote_average"]
        features["anio"] = df["release_date"].dt.year

        # Categórica: los idiomas poco frecuentes van a "otro"
        features["idioma"] = df["original_language"].where(
            df["original_language"].isin(top_idiomas), "otro"
        )

        # Un indicador 0/1 por género
        for genero in top_generos:
            columna = "gen_" + genero.lower().replace(" ", "_")
            features[columna] = generos.apply(lambda gs, g=genero: int(g in gs))

        features["revenue"] = df["revenue"]
        features = features.dropna().reset_index(drop=True)

        out_key = "processed/tmdb_features.parquet"
        buf = io.BytesIO()
        features.to_parquet(buf, index=False)
        s3.put_object(Bucket="data", Key=out_key, Body=buf.getvalue())

        print(f"💾 Features guardadas en: s3://data/{out_key}")
        return {
            "path": f"s3://data/{out_key}",
            "observations": features.shape[0],
            "columns": features.shape[1],
        }

    @task.external_python(
        task_id="split_dataset",
        python=ML_PYTHON,
        expect_airflow=False,
        multiple_outputs=True,
        outlets=[TMDB_DATA],
    )
    def split_dataset(file_path, obs, col, test_size):
        """
        Separa en entrenamiento y testeo, y registra la corrida en MLflow
        """
        import datetime  # noqa: PLC0415
        import io  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        import mlflow  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        s3 = boto3.client("s3")
        test_size = float(test_size)

        bucket, key = file_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        assert df.shape == (obs, col), (
            "⚠️ La forma del dataset no coincide con lo esperado."
        )

        print("🔀 Separando dataset en entrenamiento y prueba...")
        xx = df.drop(columns="revenue")
        y = df[["revenue"]]

        xx_train, xx_test, y_train, y_test = train_test_split(
            xx, y, test_size=test_size, random_state=42
        )

        # Rutas fijas: retrain_revenue_model lee siempre de acá
        outputs = {
            "xx_train_file_path": (xx_train, "final/train/tmdb_X_train.parquet"),
            "y_train_file_path": (y_train, "final/train/tmdb_y_train.parquet"),
            "xx_test_file_path": (xx_test, "final/test/tmdb_X_test.parquet"),
            "y_test_file_path": (y_test, "final/test/tmdb_y_test.parquet"),
        }

        paths = {}
        for name, (data, out_key) in outputs.items():
            buf = io.BytesIO()
            data.to_parquet(buf, index=False)
            s3.put_object(Bucket="data", Key=out_key, Body=buf.getvalue())
            paths[name] = f"s3://data/{out_key}"

        print("📝 Registrando corrida en MLflow...")
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("tmdb_revenue_etl")
        run_name = "etl_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "observations": obs,
                "train_rows": len(xx_train),
                "test_rows": len(xx_test),
                "test_size": test_size,
                "features": xx.columns.tolist(),
            })
            dataset = mlflow.data.from_pandas(
                xx_train.join(y_train), source=paths["xx_train_file_path"],
                targets="revenue", name="tmdb_train",
            )
            mlflow.log_input(dataset, context="training")

        print(f"✅ Train: {len(xx_train):,} películas, test: {len(xx_test):,} películas")
        return paths

    # 🧩 Encadenamiento
    raw_path = get_data()
    clean_path = clean_data(raw_path)
    data = make_features(clean_path)
    split_dataset(
        data["path"], data["observations"], data["columns"], "{{ params.test_size }}"
    )


dag = etl_tmdb_movies()

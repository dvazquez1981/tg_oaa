import datetime

from airflow.sdk import Asset, dag, task

# Virtualenv con las mismas versiones que la API (ver dockerfiles/airflow/Dockerfile)
ML_PYTHON = "/opt/ml-venv/bin/python"

# Lo actualiza etl_tmdb_movies; cuando cambia, corre este DAG
TMDB_DATA = Asset("s3://data/final/tmdb")

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="retrain_revenue_model",
    description="Reentrena el modelo y lo pone en producción si mejora al champion",
    default_args=default_args,
    schedule=[TMDB_DATA],
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=30),
    tags=["Re-Train", "TMDB"],
)
def retrain_revenue_model():

    @task.external_python(
        task_id="train_challenger",
        python=ML_PYTHON,
        expect_airflow=False,
        multiple_outputs=True,
    )
    def train_challenger():
        """
        Entrena un modelo nuevo con los hiperparámetros del champion y lo registra como challenger
        """
        import datetime  # noqa: PLC0415
        import io  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        import mlflow  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from mlflow.exceptions import MlflowException  # noqa: PLC0415
        from mlflow.models import infer_signature  # noqa: PLC0415
        from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: PLC0415
        from sklearn.metrics import mean_absolute_error  # noqa: PLC0415
        from sklearn.pipeline import Pipeline  # noqa: PLC0415
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: PLC0415
        from xgboost import XGBRegressor  # noqa: PLC0415

        s3 = boto3.client("s3")
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()
        model_name = "revenue_regressor"

        def _read(key):
            obj = s3.get_object(Bucket="data", Key=key)
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))

        xx_train = _read("final/train/tmdb_X_train.parquet")
        y_train = _read("final/train/tmdb_y_train.parquet")["revenue"]
        xx_test = _read("final/test/tmdb_X_test.parquet")
        y_test = _read("final/test/tmdb_y_test.parquet")["revenue"]

        # Si no hay champion todavía, usamos los hiperparámetros por defecto
        try:
            champion = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
            params = champion.regressor_.named_steps["modelo"].get_params()
            print("🏆 Usando hiperparámetros del champion")
        except MlflowException:
            params = {"random_state": 42, "n_jobs": -1}
            print("ℹ️ No hay champion, usando hiperparámetros por defecto")

        # Mismo pipeline que notebooks/train.ipynb
        binarias = [c for c in xx_train.columns if c.startswith("gen_")]
        preprocesador = ColumnTransformer([
            ("sesgadas", Pipeline([("log", FunctionTransformer(np.log1p)),
                                   ("escala", StandardScaler())]),
             ["budget", "popularity", "vote_count", "n_companias"]),
            ("numericas", StandardScaler(), ["runtime", "vote_average", "anio"]),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["idioma"]),
            ("binarias", "passthrough", binarias),
        ])
        model = TransformedTargetRegressor(
            regressor=Pipeline([("prep", preprocesador),
                                ("modelo", XGBRegressor(**params))]),
            func=np.log1p,
            inverse_func=np.expm1,
        )

        print("🏋️ Entrenando challenger...")
        model.fit(xx_train, y_train)

        mlflow.set_experiment("tmdb_revenue_regressor")
        run_name = "challenger_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_metric("test_mae", mean_absolute_error(y_test, model.predict(xx_test)))
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=infer_signature(xx_train, model.predict(xx_train)),
                input_example=xx_train.head(5),
                registered_model_name=model_name,
                # MLflow guarda con skops y hay que marcar XGBoost como confiable
                skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"],
            )

        version = model_info.registered_model_version
        client.set_registered_model_alias(model_name, "challenger", version)
        print(f"💾 Challenger registrado: {model_name} versión {version}")
        return {"version": version, "run_id": model_info.run_id}

    @task.external_python(
        task_id="evaluate_models",
        python=ML_PYTHON,
        expect_airflow=False,
        multiple_outputs=True,
    )
    def evaluate_models(challenger_run_id):
        """
        Calcula el MAE del champion y del challenger sobre el set de testeo
        """
        import io  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        import mlflow  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from mlflow.exceptions import MlflowException  # noqa: PLC0415
        from sklearn.metrics import mean_absolute_error  # noqa: PLC0415

        s3 = boto3.client("s3")
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        def _read(key):
            obj = s3.get_object(Bucket="data", Key=key)
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))

        xx_test = _read("final/test/tmdb_X_test.parquet")
        y_test = _read("final/test/tmdb_y_test.parquet")["revenue"]

        def _mae(alias):
            model = mlflow.sklearn.load_model(f"models:/revenue_regressor@{alias}")
            return mean_absolute_error(y_test, model.predict(xx_test))

        challenger_mae = _mae("challenger")
        client.log_metric(challenger_run_id, "test_mae_challenger", challenger_mae)
        print(f"🧪 MAE challenger: {challenger_mae:,.0f}")

        try:
            champion_mae = _mae("champion")
            client.log_metric(challenger_run_id, "test_mae_champion", champion_mae)
            print(f"🏆 MAE champion: {champion_mae:,.0f}")
        except MlflowException:
            champion_mae = None
            print("ℹ️ No hay champion para comparar")

        return {"challenger_mae": challenger_mae, "champion_mae": champion_mae}

    @task.branch(task_id="compare_models")
    def compare_models(challenger_mae, champion_mae):
        if champion_mae is None or challenger_mae < champion_mae:
            return "promote_challenger"
        return "discard_challenger"

    @task.external_python(
        task_id="promote_challenger", python=ML_PYTHON, expect_airflow=False
    )
    def promote_challenger(version):
        """
        El challenger pasa a ser el champion, que es el que usa la API
        """
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        client.delete_registered_model_alias("revenue_regressor", "challenger")
        client.set_registered_model_alias("revenue_regressor", "champion", version)
        print(f"🚀 Versión {version} en producción como champion")

    @task.external_python(
        task_id="discard_challenger", python=ML_PYTHON, expect_airflow=False
    )
    def discard_challenger(version):
        """
        El challenger no mejoró al champion, le sacamos el alias
        """
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        client.delete_registered_model_alias("revenue_regressor", "challenger")
        print(f"🛑 Versión {version} descartada, el champion sigue igual")

    # 🧩 Encadenamiento
    challenger = train_challenger()
    results = evaluate_models(challenger["run_id"])
    compare_models(results["challenger_mae"], results["champion_mae"]) >> [
        promote_challenger(challenger["version"]),
        discard_challenger(challenger["version"]),
    ]


dag = retrain_revenue_model()

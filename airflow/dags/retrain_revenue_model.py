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
    params={"n_trials": 50},
    dagrun_timeout=datetime.timedelta(hours=2),
    tags=["Re-Train", "TMDB"],
)
def retrain_revenue_model():

    @task.external_python(
        task_id="tune_hyperparameters",
        python=ML_PYTHON,
        expect_airflow=False,
        multiple_outputs=True,
    )
    def tune_hyperparameters(n_trials):
        """
        Busca los mejores hiperparámetros con Optuna (misma búsqueda que notebooks/train.ipynb)
        """
        import io  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import optuna  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: PLC0415
        from sklearn.model_selection import cross_val_score  # noqa: PLC0415
        from sklearn.pipeline import Pipeline  # noqa: PLC0415
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: PLC0415
        from xgboost import XGBRegressor  # noqa: PLC0415

        s3 = boto3.client("s3")
        n_trials = int(n_trials)

        def _read(key):
            obj = s3.get_object(Bucket="data", Key=key)
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))

        xx_train = _read("final/train/tmdb_X_train.parquet")
        y_train = _read("final/train/tmdb_y_train.parquet")["revenue"]

        # Mismo pipeline que notebooks/train.ipynb
        binarias = [c for c in xx_train.columns if c.startswith("gen_")]

        def _pipeline(estimador):
            preprocesador = ColumnTransformer([
                ("sesgadas", Pipeline([("log", FunctionTransformer(np.log1p)),
                                       ("escala", StandardScaler())]),
                 ["budget", "popularity", "vote_count", "n_companias"]),
                ("numericas", StandardScaler(), ["runtime", "vote_average", "anio"]),
                ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 ["idioma"]),
                ("binarias", "passthrough", binarias),
            ])
            return TransformedTargetRegressor(
                regressor=Pipeline([("prep", preprocesador), ("modelo", estimador)]),
                func=np.log1p,
                inverse_func=np.expm1,
            )

        def objetivo(trial):
            """MAE por validación cruzada de una configuración propuesta por Optuna."""
            candidato = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                random_state=42, n_jobs=1,
            )
            return -cross_val_score(_pipeline(candidato), xx_train, y_train, cv=5,
                                    scoring="neg_mean_absolute_error", n_jobs=-1).mean()

        estudio = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

        print(f"🔎 Buscando hiperparámetros con Optuna ({n_trials} trials)...")
        estudio.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
        print(f"✅ Mejor MAE de CV: {estudio.best_value:,.0f} con {estudio.best_params}")

        return {"best_params": estudio.best_params, "cv_mae": estudio.best_value}

    @task.external_python(
        task_id="train_challenger",
        python=ML_PYTHON,
        expect_airflow=False,
        multiple_outputs=True,
    )
    def train_challenger(best_params, cv_mae, n_trials):
        """
        Entrena un modelo nuevo con los hiperparámetros encontrados por Optuna y lo registra como challenger
        """
        import datetime  # noqa: PLC0415
        import io  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        import mlflow  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from mlflow.models import infer_signature  # noqa: PLC0415
        from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: PLC0415
        from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # noqa: PLC0415
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

        params = {**best_params, "random_state": 42, "n_jobs": -1}

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
            # Hiperparámetros encontrados por Optuna
            mlflow.log_params(best_params)
            mlflow.log_params({"n_trials": int(n_trials), "random_state": 42})

            # Métricas: MAE de validación cruzada (Optuna) y métricas sobre el set de testeo
            y_pred = model.predict(xx_test)
            mlflow.log_metrics({
                "cv_mae": cv_mae,
                "test_mae": mean_absolute_error(y_test, y_pred),
                "test_rmse": root_mean_squared_error(y_test, y_pred),
                "test_r2": r2_score(y_test, y_pred),
            })
            mlflow.set_tag("Training Info", "XGBRegressor + Optuna, target en escala log")

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
    tuned = tune_hyperparameters("{{ params.n_trials }}")
    challenger = train_challenger(tuned["best_params"], tuned["cv_mae"], "{{ params.n_trials }}")
    results = evaluate_models(challenger["run_id"])
    compare_models(results["challenger_mae"], results["champion_mae"]) >> [
        promote_challenger(challenger["version"]),
        discard_challenger(challenger["version"]),
    ]


dag = retrain_revenue_model()

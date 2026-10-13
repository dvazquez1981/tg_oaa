"""
Búsqueda de hiperparámetros del modelo de recaudación con Optuna.

Prueba combinaciones de hiperparámetros de XGBoost con validación cruzada sobre el train
del ETL, y registra cada intento en el experimento tmdb_revenue_tuning de MLflow. Con la
mejor combinación entrena un challenger, que compite contra el champion como en el
reentrenamiento.

Se ejecuta a mano, o solo cuando retrain_revenue_model encuentra que todavía no hay
champion (la primera vez que se levanta el sistema).

Acá solo se define el flujo; la lógica está en utils/tuning.py.
"""

import datetime
from airflow.sdk import Param, dag, task
from common import ML_VENV
from common.champion_challenger import compete, train_challenger
from utils import config

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="tune_revenue_model",
    description="Busca hiperparámetros con Optuna y pone el mejor modelo en producción",
    default_args=default_args,
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,   # activo desde el arranque: lo dispara retrain_revenue_model
    max_active_runs=1,
    dagrun_timeout=datetime.timedelta(hours=2),
    tags=["Tuning", "TMDB"],
    params={
        "n_trials": Param(config.TUNING_TRIALS, type="integer", minimum=1,
                          description="Cantidad de combinaciones a probar"),
        "cv_folds": Param(config.TUNING_FOLDS, type="integer", minimum=2,
                          description="Particiones de la validación cruzada"),
    },
)
def tune_revenue_model():

    @task.external_python(task_id="search_hyperparameters", **ML_VENV)
    def search_hyperparameters(n_trials, cv_folds):
        """Corre la búsqueda con Optuna, la registra en MLflow y devuelve los mejores."""
        
        from utils import tuning

        # Llegan como texto porque vienen de plantillas de Jinja
        return tuning.search_and_log(int(n_trials), int(cv_folds))

    # 🧩 Encadenamiento
    best_params = search_hyperparameters("{{ params.n_trials }}", "{{ params.cv_folds }}")
    challenger = train_challenger(best_params, "optuna")
    compete(challenger)


tune_revenue_model()

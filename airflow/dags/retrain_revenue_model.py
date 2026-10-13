"""
Reentrenamiento del modelo de recaudación con el esquema champion/challenger.

Se ejecuta solo cada vez que etl_tmdb_movies actualiza los datos (asset TMDB_DATA).

- Si ya hay champion: entrena un challenger con sus mismos hiperparámetros sobre los datos
  nuevos, compara los dos sobre el mismo set de testeo y promueve al challenger solo si
  tiene menor error.
- Si todavía no hay champion (la primera vez): dispara tune_revenue_model, que busca los
  hiperparámetros con Optuna y pone en producción el primer modelo.

Acá solo se define el flujo; la lógica está en utils/model.py y common/champion_challenger.py.
"""

import datetime
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Asset, dag, task
from common import ML_VENV
from common.champion_challenger import compete, train_challenger
from utils import config


TMDB_DATA = Asset(config.TMDB_ASSET_URI)

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
    is_paused_upon_creation=False,
    max_active_runs=1,
    dagrun_timeout=datetime.timedelta(minutes=30),
    tags=["Re-Train", "TMDB"],
)
def retrain_revenue_model():

    @task.branch_external_python(task_id="check_champion", **ML_VENV)
    def check_champion():
        """Elige el camino: reentrenar si hay champion, o buscar hiperparámetros si no."""
        
        from utils import model

        if model.has_champion():
            print("🏆 Hay champion: se reentrena con sus hiperparámetros")
            return "train_challenger"
        print("ℹ️ No hay champion: se dispara la búsqueda de hiperparámetros")
        return "trigger_tuning"

    trigger_tuning = TriggerDagRunOperator(
        task_id="trigger_tuning",
        trigger_dag_id="tune_revenue_model",
    )

    # 🧩 Encadenamiento
    challenger = train_challenger(None, "champion")
    check_champion() >> [challenger, trigger_tuning]
    compete(challenger)


retrain_revenue_model()

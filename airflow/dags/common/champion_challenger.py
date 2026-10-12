"""
Entrenamiento de un challenger y duelo contra el champion, compartido por los DAGs
retrain_revenue_model y tune_revenue_model.

El champion es la versión del modelo en producción (la que usa la API) y el challenger un
modelo recién entrenado. Los dos se evalúan sobre el mismo set de testeo y el challenger
pasa a ser champion solo si tiene menor error. La lógica está en utils/model.py.

Las tareas se definen acá una vez y cada DAG las llama dentro de su propia definición.
"""

from airflow.sdk import task
from common import ML_VENV


@task.external_python(task_id="train_challenger", multiple_outputs=True, **ML_VENV)
def train_challenger(hyperparams, origin):
    """Entrena un modelo y lo registra con el alias challenger.

    Args:
        hyperparams: hiperparámetros a usar, o None para copiar los del champion actual.
            (No se puede llamar "params": es un nombre reservado de Airflow.)
        origin: de dónde salen los hiperparámetros; queda como tag en MLflow.
    """

    from utils import model

    if hyperparams is None:
        hyperparams = model.champion_params()
    return model.train_and_register(hyperparams, origin)


@task.external_python(task_id="evaluate_models", **ML_VENV)
def evaluate_models(challenger_run_id):
    """Calcula el MAE del challenger y del champion sobre el mismo set de testeo.

    Devuelve {"challenger_mae": float, "champion_mae": float o None si no hay champion}.
    Sin multiple_outputs a propósito: Airflow 3.0.2 falla al guardar en XCom un None
    suelto, y dentro de un diccionario no hay problema.
    """

    from utils import config, model

    _, _, x_test, y_test = model.load_split()
    client = model.mlflow_client()
    challenger_mae = model.evaluate(model.load_alias(config.CHALLENGER), x_test, y_test)
    client.log_metric(challenger_run_id, "test_mae_challenger", challenger_mae)
    print(f"🧪 MAE challenger: {challenger_mae:,.0f}")

    champion = model.load_alias(config.CHAMPION)
    if champion is None:
        print("ℹ️ No hay champion para comparar")
        return {"challenger_mae": challenger_mae, "champion_mae": None}

    champion_mae = model.evaluate(champion, x_test, y_test)
    client.log_metric(challenger_run_id, "test_mae_champion", champion_mae)
    print(f"🏆 MAE champion: {champion_mae:,.0f}")
    return {"challenger_mae": challenger_mae, "champion_mae": champion_mae}


@task.branch(task_id="compare_models")
def compare_models(maes):
    """Elige el camino: promover si el challenger tiene menor error o no hay champion."""

    challenger_mae, champion_mae = maes["challenger_mae"], maes["champion_mae"]
    if champion_mae is None or challenger_mae < champion_mae:
        return "promote_challenger"
    return "discard_challenger"


@task.external_python(task_id="promote_challenger", **ML_VENV)
def promote_challenger(version):
    """El challenger pasa a ser el champion, que es el modelo que usa la API."""

    from utils import model

    model.set_champion(version)
    print(f"🚀 Versión {version} en producción como champion")


@task.external_python(task_id="discard_challenger", **ML_VENV)
def discard_challenger(version):
    """El challenger no mejoró al champion: se le saca el alias."""

    from utils import model

    model.drop_challenger()
    print(f"🛑 Versión {version} descartada, el champion sigue igual")


def compete(challenger):
    """Encadena evaluación, comparación y promoción o descarte de un challenger.

    Args:
        challenger: salida de train_challenger (versión y run_id del modelo registrado).
    """
    
    maes = evaluate_models(challenger["run_id"])
    compare_models(maes) >> [
        promote_challenger(challenger["version"]),
        discard_challenger(challenger["version"]),
    ]

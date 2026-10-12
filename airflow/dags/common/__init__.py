"""
Tareas de Airflow que comparten varios DAGs.

A diferencia de utils/, este paquete sí importa Airflow: lo usan los archivos de los DAGs,
que corren en el Python de Airflow. Está en .airflowignore porque no define DAGs.
"""

from utils import config

# Argumentos de las tareas que corren en el virtualenv de ML, sin Airflow
# (ver dockerfiles/airflow/Dockerfile). Solo se ejecuta el código de la función, por eso
# utils se importa adentro. El decorador tiene que quedar escrito como
# @task.external_python: Airflow lo busca por ese nombre para sacarlo del código.
ML_VENV = {"python": config.ML_PYTHON, "expect_airflow": False}

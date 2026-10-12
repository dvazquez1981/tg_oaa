"""
Lógica compartida por los DAGs del proyecto.

Los módulos de acá corren dentro de las tareas, en el virtualenv /opt/ml-venv, que no
tiene Airflow instalado. Por eso no importan nada de Airflow y se pueden probar con pytest.

- config: rutas, nombres y constantes. Es el único que importan los archivos de los DAGs
  al principio, porque solo usa la librería estándar.
- s3: leer y escribir DataFrames en parquet sobre S3.
- data: descarga, limpieza, features y split del dataset de TMDB.
- model: pipeline del modelo y manejo de champion/challenger en MLflow.
"""

# Operaciones de Aprendizaje Automático I — Trabajo Práctico Final

**Especialización en Inteligencia Artificial (CEIA) — FIUBA**

Implementación en un ambiente productivo simulado del modelo desarrollado en Aprendizaje de Máquina I: un regresor que **estima la recaudación de una película** a partir de sus datos de TMDB.

El sistema usa herramientas de MLOps: orquestación con **Apache Airflow**, ciclo de vida de modelos con **MLflow**, un Data Lake compatible con **S3** y una **API REST** con FastAPI que sirve el modelo. Todo corre en contenedores **Docker** y, al levantarlo por primera vez, se entrena y se pone en producción un modelo **sin intervención manual**.

Proyecto basado en el template [amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml) de la cátedra.

## Integrantes

* Luis Jose Paredes Ramirez
* Diego Vazquez
* Hernando Scheidl
* Nicolás Lastra

---

## Contenido

1. [Puesta en marcha rápida](#puesta-en-marcha-rápida)
2. [El problema y el modelo](#el-problema-y-el-modelo)
3. [Arquitectura](#arquitectura)
4. [Pipeline automático: los DAGs](#pipeline-automático-los-dags)
5. [Experimentos y modelos en MLflow](#experimentos-y-modelos-en-mlflow)
6. [API del modelo](#api-del-modelo)
7. [Estructura del repositorio](#estructura-del-repositorio)
8. [Entorno local, notebooks y tests](#entorno-local-notebooks-y-tests)
9. [Operación del sistema](#operación-del-sistema)
10. [Decisiones de diseño](#decisiones-de-diseño)
11. [Cambios respecto al template de la cátedra](#cambios-respecto-al-template-de-la-cátedra)
12. [Problemas frecuentes](#problemas-frecuentes)

---

## Puesta en marcha rápida

```bash
git clone <url-del-repo>
cd <carpeta-del-repo>
docker compose --profile all up -d
```

La primera vez tarda varios minutos en construir las imágenes. Después **no hay que ejecutar nada a mano**: Airflow descarga el dataset, busca hiperparámetros, entrena y pone un modelo en producción solo. En 5 a 10 minutos (según la conexión y la PC), la API está lista:

```bash
curl http://localhost:8800/health
# {"status":"ok","model_name":"revenue_regressor","model_alias":"champion","model_version":"1"}
```

| Servicio | Para qué | URL | Usuario / clave |
|---|---|---|---|
| Apache Airflow | Ver y ejecutar los DAGs | http://localhost:8080 | `airflow` / `airflow` |
| MLflow | Experimentos y Model Registry | http://localhost:5001 | — |
| RustFS (consola) | Explorar los buckets `data` y `mlflow` | http://localhost:9001 | `minio` / `minio123` |
| API (documentación) | Probar predicciones | http://localhost:8800/docs | — |

Para verificar el estado de los contenedores:

```bash
docker compose --profile all ps
```

Todos deben quedar `healthy`. `create_s3_buckets` y `airflow-init` son tareas de inicialización: es normal que terminen en `Exited (0)`.

---

## El problema y el modelo

### Datos

Se usa el dataset de Kaggle [Full TMDB Movies Dataset 2024 (1M Movies)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies), con información de películas extraída de la API de [The Movie Database (TMDB)](https://www.themoviedb.org/). Es público y se descarga sin credenciales.

TMDB es una base colaborativa: la gran mayoría de las películas no tiene cargados el presupuesto o la recaudación. Por eso el ETL se queda solo con las películas **estrenadas** que tienen **recaudación, presupuesto, duración y votos mayores a cero**, fecha de estreno y al menos un género. En la última corrida:

| Etapa | Películas |
|---|---|
| Dataset crudo | 1.514.263 |
| Después de limpiar y armar features | 10.542 |
| Entrenamiento (80 %) | 8.433 |
| Testeo (20 %) | 2.109 |

El detalle de cuántas películas descarta cada filtro queda en el log de la tarea `clean_data`. El análisis completo está en `notebooks/00_EDA.ipynb`.

### Features

| Grupo | Columnas | Tratamiento en el pipeline |
|---|---|---|
| Sesgadas | `budget`, `popularity`, `vote_count`, `n_companias` (cantidad de productoras) | `log1p` + estandarización |
| Numéricas | `runtime`, `vote_average`, `anio` | estandarización |
| Categórica | `idioma`: los 8 idiomas más frecuentes; el resto pasa a `"otro"` | one-hot |
| Binarias | `gen_*`: una columna 0/1 por cada uno de los 12 géneros más frecuentes | sin cambios |

### Modelo

Un **XGBoost** dentro de un pipeline de scikit-learn (`TransformedTargetRegressor`) que entrena sobre `log(1 + revenue)` y deshace el logaritmo al predecir, así que las predicciones salen directamente en dólares. El target es muy asimétrico (pocas películas recaudan muchísimo), y trabajar en escala logarítmica estabiliza el entrenamiento.

XGBoost fue el mejor modelo en la comparación del EDA, contra un baseline, Ridge, KNN, SVR, árbol de decisión y Random Forest. Sus hiperparámetros se buscan con **Optuna** y se elige el modelo por **MAE** (error absoluto medio, en dólares).

---

## Arquitectura

Supuesto de la consigna: trabajamos para **ML Models and something more Inc.**, que ofrece modelos mediante una REST API y cuenta internamente con Airflow, MLflow y un Data Lake en S3.

```
                        ┌──────────────────────────┐
                        │   Apache Airflow (3.0)   │
                        │ scheduler · worker · api │──── cola de tareas ──► Valkey (Redis)
                        └────────────┬─────────────┘
                 lee/escribe datos   │   registra runs y modelos
                        ┌────────────┴─────────────┐
                        ▼                          ▼
          ┌──────────────────────┐     ┌──────────────────────┐
          │   RustFS (S3)        │◄────│   MLflow (3.16)      │
          │   s3://data          │     │   tracking + registry│
          │   s3://mlflow        │     └──────────┬───────────┘
          └──────────────────────┘                │ metadata
                        ▲                         ▼
                        │              ┌──────────────────────┐
                        │              │   PostgreSQL         │
                        │              │   airflow · mlflow_db│
                        │              └──────────────────────┘
          ┌─────────────┴────────┐
          │   FastAPI            │  carga el modelo champion desde MLflow
          │   REST API           │
          └──────────────────────┘
```

| Servicio | Rol |
|---|---|
| **Apache Airflow 3.0.2** | Orquesta el pipeline. Corre en varios contenedores: `airflow-apiserver` (interfaz web y API), `airflow-scheduler` (decide qué corre y cuándo), `airflow-dag-processor` (lee los DAGs), `airflow-worker` (ejecuta las tareas) y `airflow-triggerer`. |
| **Valkey** | Cola de mensajes de Celery: el scheduler deja las tareas y los workers las toman. Solo en la red interna. |
| **RustFS** | Data Lake compatible con S3. Bucket `data` (datasets de cada etapa) y bucket `mlflow` (modelos y artefactos). |
| **MLflow 3.16** | Registra experimentos (parámetros, métricas, datasets) y versiona los modelos en el Model Registry. |
| **PostgreSQL** | Metadata de Airflow (base `airflow`) y de MLflow (base `mlflow_db`). |
| **FastAPI** | Sirve el modelo con alias `champion`. |

Los puertos y credenciales se configuran en el archivo `.env`.

---

## Pipeline automático: los DAGs

```
etl_tmdb_movies ──(asset s3://data/final/tmdb)──► retrain_revenue_model
 (una vez, al arrancar)                                │
                                ¿hay champion?  ───────┤
                                     no ▼              ▼ sí
                            tune_revenue_model   reentrena con los
                            (búsqueda Optuna)    hiperparámetros del champion
                                     │                 │
                                     └──► duelo challenger vs. champion ──► champion
```

| DAG | Se ejecuta | Qué hace |
|---|---|---|
| `etl_tmdb_movies` | Una vez al arrancar (`@once`), y después a mano para actualizar el dataset | Descarga, limpia, arma features y separa train/test. |
| `retrain_revenue_model` | Solo, cada vez que termina el ETL | Si hay champion, reentrena con sus hiperparámetros. Si no, dispara la búsqueda. |
| `tune_revenue_model` | Cuando no hay champion, o a mano | Busca hiperparámetros con Optuna y entrena con los mejores. |

Los tres DAGs arrancan **activos** (`is_paused_upon_creation=False`), así el sistema funciona sin tocar nada. Cualquier DAG nuevo que se agregue arranca pausado, por la configuración general del `docker-compose.yaml`.

### `etl_tmdb_movies`

| Tarea | Qué hace | Guarda en |
|---|---|---|
| `obtain_original_data` | Crea los buckets si faltan, descarga el CSV de Kaggle (~600 MB) y se queda con 13 columnas. Borra el CSV del contenedor al terminar. | `s3://data/raw/tmdb_movies.parquet` |
| `clean_data` | Aplica los filtros y saca duplicados. Imprime cuántas películas quedan después de cada filtro. | `s3://data/processed/tmdb_movies_clean.parquet` |
| `make_features` | Arma la matriz de features y el target. | `s3://data/processed/tmdb_features.parquet` |
| `split_dataset` | Separa train/test (parámetro `test_size`, por defecto 0.2), registra la corrida en MLflow y **actualiza el asset** que dispara el reentrenamiento. | `s3://data/final/train/` y `s3://data/final/test/` |

Las tareas se pasan entre sí **rutas de S3**, no datos: los datos viajan por el Data Lake y Airflow (XCom) solo transporta las direcciones.

### `retrain_revenue_model`

Programado por el **asset** `s3://data/final/tmdb`: corre cada vez que el ETL termina bien.

1. `check_champion` (bifurcación): consulta el Model Registry.
   - **No hay champion** → `trigger_tuning` dispara `tune_revenue_model` y el DAG termina.
   - **Hay champion** → sigue el reentrenamiento.
2. `train_challenger`: entrena un modelo nuevo **con los mismos hiperparámetros del champion** sobre los datos actuales, lo registra y le pone el alias `challenger`.
3. Duelo champion/challenger (ver abajo).

### `tune_revenue_model`

1. `search_hyperparameters`: búsqueda con Optuna (sampler TPE, semilla fija) usando **validación cruzada sobre el set de entrenamiento**. El de testeo no se usa, para no sesgar la comparación posterior. Cada intento queda registrado en MLflow. Parámetros configurables al ejecutarlo: `n_trials` (por defecto 50) y `cv_folds` (por defecto 5). El espacio de búsqueda es el mismo que el de `notebooks/01_TRAIN.ipynb`.
2. `train_challenger`: entrena con la mejor combinación.
3. Duelo champion/challenger.

### Duelo champion/challenger

Compartido por los dos DAGs anteriores (`airflow/dags/common/champion_challenger.py`):

| Tarea | Qué hace |
|---|---|
| `evaluate_models` | Calcula el MAE del challenger y del champion sobre **el mismo set de testeo**. |
| `compare_models` | Bifurcación: si el challenger tiene **menor** MAE, o si no había champion, sigue `promote_challenger`; si no, `discard_challenger`. |
| `promote_challenger` | El alias `champion` pasa a la versión nueva. La API la toma sola. |
| `discard_challenger` | Le saca el alias `challenger`; el champion sigue igual. |

En cada corrida, una de las dos últimas tareas queda en *skipped*: es el comportamiento esperado de una bifurcación.

Un empate cuenta como derrota del challenger. Si se reentrena con los mismos datos, el resultado es idéntico (datos, división y semillas son fijos) y el champion se mantiene. El challenger le gana cuando hay datos nuevos o mejores hiperparámetros.

---

## Experimentos y modelos en MLflow

En http://localhost:5001:

| Experimento | Contenido |
|---|---|
| `tmdb_revenue_etl` | Una corrida por ETL: cantidad de filas, tamaño de train/test, columnas y el dataset de entrenamiento registrado como *input* (esquema, origen y hash). |
| `tmdb_revenue_tuning` | **Búsqueda de hiperparámetros.** Una corrida padre `optuna_<fecha>` con los mejores hiperparámetros y `best_cv_mae`, y una corrida hija por cada intento con sus hiperparámetros y su `cv_mae`. Seleccionando varios intentos y usando **Compare** se ve el efecto de cada hiperparámetro. |
| `tmdb_revenue_regressor` | Una corrida por cada challenger entrenado: hiperparámetros, `test_mae`, `test_mae_challenger`, `test_mae_champion` y el modelo. El tag `origen_hiperparametros` indica si salieron de `optuna` o del `champion`. |

**Model Registry** (en la barra izquierda, *Model registry*): el modelo `revenue_regressor` con todas sus versiones. La que tiene el alias **`@champion`** es la que está en producción. Las versiones descartadas quedan registradas sin alias, como historial.

---

## API del modelo

Documentación interactiva en http://localhost:8800/docs, con un ejemplo listo para probar con **Try it out**.

| Endpoint | Qué hace |
|---|---|
| `POST /predict` | Recibe los datos de una película y devuelve la recaudación estimada en USD y la versión del modelo que la calculó. |
| `GET /health` | Estado del servicio (`ok` o `sin_modelo`) y versión cargada. |
| `GET /` | Mensaje de bienvenida. |

### Ejemplo

```bash
curl -X POST http://localhost:8800/predict -H "Content-Type: application/json" -d '{
  "budget": 25000000, "popularity": 16.3, "vote_count": 3427, "vote_average": 7.2,
  "runtime": 143, "release_year": 2006, "original_language": "en",
  "genres": ["Drama", "Crime"], "n_production_companies": 6
}'
```

```json
{"revenue": 51018448.0, "model_name": "revenue_regressor", "model_version": "1"}
```

El valor exacto depende del modelo entrenado.

### Campos de entrada

| Campo | Tipo | Validación |
|---|---|---|
| `budget` | entero | presupuesto en USD, mayor a 0 |
| `popularity` | número | popularidad en TMDB, ≥ 0 |
| `vote_count` | entero | cantidad de votos, mayor a 0 |
| `vote_average` | número | puntaje, entre 0 (excluido) y 10 |
| `runtime` | entero | duración en minutos, mayor a 0 |
| `release_year` | entero | entre 1850 y 2100 |
| `original_language` | texto | código ISO 639-1, por ejemplo `"en"` |
| `genres` | lista de textos | nombres de TMDB, por ejemplo `["Drama", "Science Fiction"]`; opcional |
| `n_production_companies` | entero | ≥ 0; opcional, por defecto 1 |

Si un dato no cumple la validación, la API responde **422** indicando qué campo está mal. Los idiomas que el modelo no conoce se tratan como `"otro"` y los géneros desconocidos se ignoran, igual que en el entrenamiento.

### Comportamiento

- **Sin modelo**: si todavía no hay champion, la API arranca igual. `/predict` responde **503** y `/health` informa `sin_modelo`.
- **Modelo nuevo**: no hace falta reiniciarla. Cada 30 segundos (`MODEL_CHECK_SECONDS` en el `docker-compose.yaml`) revisa a qué versión apunta el alias `champion` y la carga si cambió.
- **MLflow caído**: sigue respondiendo con el modelo que tiene cargado.

---

## Estructura del repositorio

```
.
├── airflow/                      # Se monta en vivo en los contenedores de Airflow
│   ├── dags/                     # DAGs del proyecto (solo definen el flujo)
│   │   ├── .airflowignore        # Carpetas sin DAGs: utils/ y common/
│   │   ├── etl_tmdb_movies.py
│   │   ├── retrain_revenue_model.py
│   │   ├── tune_revenue_model.py
│   │   ├── common/               # Tareas de Airflow compartidas
│   │   │   ├── __init__.py       # ML_VENV: argumentos de las tareas del venv de ML
│   │   │   └── champion_challenger.py
│   │   └── utils/                # Lógica de las tareas, sin Airflow
│   │       ├── config.py         # Rutas, nombres y constantes
│   │       ├── s3.py             # Lectura y escritura de parquet en S3
│   │       ├── data.py           # Descarga, limpieza, features y split
│   │       ├── model.py          # Pipeline del modelo y champion/challenger
│   │       └── tuning.py         # Búsqueda de hiperparámetros con Optuna
│   ├── secrets/                  # variables.yaml y connections.yaml de Airflow
│   ├── plugins/
│   ├── config/                   # Generado por Airflow (ignorado por git)
│   └── logs/                     # Logs de las tareas (ignorado por git)
├── dockerfiles/
│   ├── airflow/                  # apache/airflow:3.0.2 + venv de ML
│   │   ├── requirements.txt      # Python de Airflow
│   │   └── requirements-ml.txt   # Venv /opt/ml-venv de las tareas de ML
│   ├── fastapi/                  # API: app.py (endpoints), model.py (modelo), schema.py (entrada/salida)
│   ├── mlflow/                   # Servidor de MLflow
│   └── postgres/                 # Postgres + creación de la base mlflow_db
├── notebooks/
│   ├── 00_EDA.ipynb              # Análisis exploratorio y comparación de modelos
│   └── 01_TRAIN.ipynb            # Entrenamiento interactivo con Optuna
├── tests/                        # Tests con pytest
├── docs/arquitectura.txt
├── docker-compose.yaml           # Definición de todos los servicios
├── pyproject.toml                # Dependencias del entorno local (uv)
├── uv.lock                       # Versiones exactas del entorno local
└── .env                          # Puertos y credenciales (valores de ejemplo)
```

---

## Entorno local, notebooks y tests

### Entorno local

Se maneja con [uv](https://docs.astral.sh/uv/). Las librerías del modelo tienen **las mismas versiones exactas** que los contenedores, así un modelo entrenado en un lado se puede cargar en el otro.

```bash
uv sync
uv run python -m ipykernel install --user --name ceia-mlops --display-name "ceia-mlops"
```

### Notebooks

Abrirlas con Jupyter o VS Code y elegir el kernel `ceia-mlops`.

- `00_EDA.ipynb`: análisis exploratorio, filtros y comparación de modelos.
- `01_TRAIN.ipynb`: entrenamiento con búsqueda de hiperparámetros, de forma interactiva. `tune_revenue_model` hace lo mismo dentro de Airflow, con el mismo espacio de búsqueda.

Para conectarse a los servicios desde una notebook, se usan los puertos publicados en la PC:

```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["AWS_ENDPOINT_URL_S3"] = "http://localhost:9000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

import mlflow
mlflow.set_tracking_uri("http://localhost:5001")
```

### Tests

```bash
uv run pytest
```

Prueban con datos sintéticos, en segundos y sin levantar los servicios:

| Archivo | Qué prueba |
|---|---|
| `test_data.py` | Limpieza, features y split del ETL. |
| `test_model.py` | El pipeline del modelo y los nombres de pasos que usa la API. |
| `test_tuning.py` | La búsqueda de hiperparámetros: registra cada intento, es reproducible y sus resultados sirven para entrenar. |
| `test_api.py` | Los endpoints, las validaciones, la recarga del champion, el caso de MLflow caído, y que la API arme las features **igual que el ETL**. |

---

## Operación del sistema

### Actualizar el dataset

Ejecutar `etl_tmdb_movies` desde ▶ *Trigger* en Airflow. Al terminar dispara solo el reentrenamiento, y si el modelo nuevo es mejor, la API lo empieza a usar en hasta 30 segundos.

### Volver a buscar hiperparámetros

Ejecutar `tune_revenue_model` desde ▶ *Trigger*. En el formulario se pueden cambiar `n_trials` (por ejemplo, 150, como en la notebook) y `cv_folds`.

### Ver logs

- **De una tarea**: en Airflow, entrar al DAG, hacer clic en el cuadrito de la tarea y abrir la pestaña **Logs**. Si la tarea se reintentó, arriba hay un selector de intento. También están en `airflow/logs/dag_id=.../run_id=.../task_id=.../attempt=N.log`.
- **De un servicio**: `docker compose logs -f fastapi` (o `mlflow`, `airflow-worker`, `airflow-scheduler`, etc.).

### Apagar y reiniciar

```bash
docker compose --profile all down            # apaga; conserva datos y modelos
docker compose --profile all up -d           # vuelve a levantar
docker compose --profile all up -d --build   # si cambiaron Dockerfiles o requirements
```

### Empezar de cero

```bash
docker compose down --volumes                # BORRA bases de datos, buckets y modelos
docker compose --profile all up -d --build
```

Al volver a levantar, el pipeline automático corre de nuevo desde el principio.

---

## Decisiones de diseño

**Virtualenv aparte para las tareas de ML.** Airflow 3.0.2 está probado con versiones viejas de las librerías de datos (por ejemplo, pandas 2.1 y numpy 1.26), y el modelo usa pandas 3, numpy 2.5, scikit-learn 1.9, XGBoost y MLflow 3. Para no mezclarlas, la imagen de Airflow tiene un segundo Python en `/opt/ml-venv` (`requirements-ml.txt`) con las mismas versiones que la API. Las tareas de ML corren ahí con `@task.external_python`. Consecuencias:
- los imports van **dentro** de cada tarea, porque Airflow solo ejecuta el código de la función en ese Python;
- dentro de esas tareas no se puede usar `Variable.get()` ni hooks de Airflow;
- en los logs aparecen avisos de que el venv no tiene Airflow (ver [Problemas frecuentes](#problemas-frecuentes)).

**DAGs livianos, lógica en `utils/`.** Los archivos de los DAGs solo definen el flujo. La lógica vive en `airflow/dags/utils/`, que no depende de Airflow y por eso se testea con pytest. Las tareas encuentran `utils` gracias a `PYTHONPATH=/opt/airflow/dags`, definido en el `docker-compose.yaml`. Las tareas de Airflow compartidas entre DAGs están en `common/`.

**Datos por S3, rutas por XCom.** Cada etapa del ETL guarda su resultado en parquet en el Data Lake y le pasa a la siguiente solo la ruta. Los resultados intermedios quedan disponibles para inspeccionar y XCom (que se guarda en Postgres) no se llena de datos.

**DAGs encadenados por asset.** El ETL no sabe qué DAG viene después: solo anuncia que actualizó `s3://data/final/tmdb`. Cualquier DAG que dependa de esos datos se suscribe con `schedule=[asset]`.

**Champion/challenger con alias de MLflow.** La API no sabe de versiones: siempre pide `revenue_regressor@champion`. Promover un modelo es mover el alias, sin redeploy ni reinicio.

**Configuración en un solo lugar.** Rutas, nombres de experimentos y del modelo están en `utils/config.py`, que usan todos los DAGs. Así, por ejemplo, la URI del asset es idéntica en el DAG que lo produce y en el que lo consume.

---

## Cambios respecto al template de la cátedra

- **MinIO → RustFS.** En septiembre de 2026 MinIO retiró sus imágenes públicas de Docker Hub y quay.io, por lo que el servicio `s3` no podía descargarse. Se reemplazó por RustFS, compatible con la API de S3. El servicio mantiene el nombre `s3` y las mismas credenciales, así que el resto del sistema no cambia.
- **Creación de buckets con AWS CLI.** El contenedor `create_s3_buckets` usa la imagen oficial `amazon/aws-cli` en lugar del cliente `mc` de MinIO. Además, el ETL crea los buckets si faltan.
- **Driver de Postgres explícito en MLflow.** La conexión del servidor de MLflow usa `postgresql+psycopg2://`. Con las versiones actuales de SQLAlchemy, el driver por defecto pasó a ser `psycopg` (v3), que no está instalado.
- **Versiones fijadas.** Las librerías que guardan y cargan el modelo tienen la misma versión exacta en el entorno local y en todos los contenedores.
- **Virtualenv de ML en la imagen de Airflow** (`/opt/ml-venv`), ver [Decisiones de diseño](#decisiones-de-diseño).
- **`MLFLOW_TRACKING_URI` y `PYTHONPATH`** definidos en el `docker-compose.yaml` para Airflow, y la configuración del modelo para FastAPI.
- **La API no depende de Airflow.** El servicio `fastapi` solo espera a MLflow, y su healthcheck usa `/health`.
- **Arranque automático.** Al levantar el sistema por primera vez, los DAGs corren solos hasta dejar un modelo en producción.

---

## Problemas frecuentes

**En los logs de las tareas aparece `No module named 'pendulum'` o `No package metadata was found for apache-airflow`.**
Es normal. Antes de ejecutar cada tarea, Airflow revisa si el venv de ML tiene Airflow instalado; como no lo tiene (a propósito), deja esos avisos en rojo y sigue. El log real de la tarea empieza después de la línea `Executing cmd: /opt/ml-venv/bin/python ...`.

**No arrancó nada solo / un DAG aparece pausado.**
`is_paused_upon_creation` solo actúa la **primera vez** que Airflow ve un DAG; después manda el estado guardado en Postgres. Si la base ya existía, activar los DAGs con el interruptor, o empezar de cero con `docker compose down --volumes`. Además, `@once` corre una sola vez por base de datos.

**El reentrenamiento no se disparó al terminar el ETL.**
Si `retrain_revenue_model` está pausado cuando el ETL actualiza el asset, Airflow **descarta** ese evento: activarlo después no lo recupera. Activarlo y ejecutarlo a mano, o volver a correr el ETL.

**Error `NoSuchBucket`.**
Faltan los buckets, por ejemplo si se borraron desde la consola de RustFS. El ETL los crea solo; si el error sale en otro DAG, ejecutar primero el ETL o `docker compose --profile all start create_s3_buckets`.

**La API responde 503.**
Todavía no hay modelo con alias `champion`. Esperar a que termine el pipeline automático, o revisar en Airflow si falló alguna tarea.

**Un DAG aparece con error de import (cartel rojo arriba en Airflow).**
Revisar que estén todos los archivos de `airflow/dags/`, en especial `common/__init__.py`. Desde la terminal: `docker compose exec airflow-scheduler airflow dags list-import-errors`.

**Cambié un DAG o `utils/` y no se refleja.**
Las carpetas se montan en vivo: Airflow relee los archivos modificados en unos segundos, pero puede tardar hasta 5 minutos en descubrir archivos **nuevos**. Si cambiaste un `Dockerfile` o un `requirements`, hace falta `docker compose --profile all up -d --build`.

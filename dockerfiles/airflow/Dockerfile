FROM apache/airflow:3.0.2

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Virtualenv aparte para las tareas de ML (@task.external_python en los DAGs).
# Tiene las mismas versiones que la API, así los modelos que entrenan los DAGs se pueden
# cargar en FastAPI, sin mezclarse con las dependencias del propio Airflow.
COPY requirements-ml.txt /requirements-ml.txt
RUN python -m venv /opt/ml-venv \
  && /opt/ml-venv/bin/pip install --no-cache-dir --upgrade pip \
  && /opt/ml-venv/bin/pip install --no-cache-dir -r /requirements-ml.txt

USER airflow
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt

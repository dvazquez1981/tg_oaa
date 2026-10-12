"""
Lectura y escritura de DataFrames en parquet sobre S3 (RustFS).

boto3 toma el endpoint y las credenciales de las variables de entorno AWS_* que define
el docker-compose.
"""

import io
import boto3
import pandas as pd


def _bucket_and_key(path: str) -> tuple[str, str]:
    """Separa "s3://bucket/carpeta/archivo" en ("bucket", "carpeta/archivo")."""
    
    bucket, key = path.removeprefix("s3://").split("/", 1)
    return bucket, key


def ensure_buckets(*buckets: str) -> None:
    """Crea los buckets que no existan.

    Normalmente los crea el contenedor create_s3_buckets al levantar Docker; esto cubre
    el caso de que se hayan borrado a mano después.
    """

    from botocore.exceptions import ClientError

    client = boto3.client("s3")
    for bucket in buckets:
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError:
            client.create_bucket(Bucket=bucket)
            print(f"🪣 Bucket s3://{bucket} creado")


def read_parquet(path: str) -> pd.DataFrame:
    """Lee un parquet de S3.

    Args:
        path: ruta con la forma s3://bucket/key.

    Returns:
        El DataFrame guardado en esa ruta.
    """

    bucket, key = _bucket_and_key(path)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def write_parquet(df: pd.DataFrame, path: str) -> str:
    """Guarda un DataFrame como parquet en S3, sin pasar por el disco.

    Args:
        df: datos a guardar. El índice no se guarda.
        path: ruta destino con la forma s3://bucket/key.

    Returns:
        La misma ruta, para devolverla desde la tarea y pasarla a la siguiente.
    """

    bucket, key = _bucket_and_key(path)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"💾 {len(df):,} filas guardadas en {path}")
    return path

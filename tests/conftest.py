"""Configuración común de pytest."""
import sys

import pytest


def pytest_configure(config):
    # En macOS XGBoost necesita libomp, que no se instala con uv sync.
    # Sin esto, pytest muestra un traceback largo por cada archivo de tests.
    if sys.platform != "darwin":
        return
    try:
        import xgboost  # noqa: F401
    except Exception as e:
        if "libomp" not in str(e):
            raise
        pytest.exit(
            "XGBoost no encuentra libomp (el runtime de OpenMP). "
            "Instalarlo con: brew install libomp",
            returncode=1,
        )

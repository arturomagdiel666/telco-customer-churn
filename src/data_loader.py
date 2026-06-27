"""
data_loader.py
==============
Módulo responsable de la **Fase 2** del proyecto: obtención y carga del dataset.

Este módulo encapsula la lectura del archivo CSV original del dataset
"Telco Customer Churn" (Kaggle - IBM) y devuelve un DataFrame de pandas
listo para las fases posteriores de auditoría de calidad y limpieza.

Decisiones de diseño documentadas:
- Se separa la carga del resto del pipeline para que sea fácilmente testeable.
- No se aplican transformaciones aquí: este módulo solo *lee*. La limpieza
  vive en `data_cleaning.py`. Esto respeta el principio de responsabilidad
  única (SRP) y facilita que un estudiante entienda qué hace cada paso.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# Columnas esperadas según el diccionario de datos publicado en Kaggle.
# Validamos su presencia para detectar archivos corruptos o versiones distintas.
EXPECTED_COLUMNS: list[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset crudo desde un archivo CSV.

    Parameters
    ----------
    csv_path : str | Path
        Ruta al archivo CSV original (`data/raw/CustomerChurn.csv`).

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos crudos, sin ninguna transformación.

    Raises
    ------
    FileNotFoundError
        Si la ruta proporcionada no existe.
    ValueError
        Si el dataset no contiene las columnas esperadas (validación de esquema).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {csv_path}")

    # `keep_default_na=True` detecta automáticamente NaN, NA, etc.
    # No forzamos dtype todavía: queremos ver cómo lo infiere pandas
    # para detectar luego problemas como `TotalCharges` cargado como string.
    df = pd.read_csv(csv_path)

    # Validación de esquema: el dataset debe traer exactamente estas columnas.
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"El dataset no contiene las columnas esperadas. Faltan: {missing}"
        )

    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """
    Devuelve un resumen estructural rápido del dataset (filas, columnas, dtypes,
    consumo de memoria). Útil para la primera celda de inspección académica.
    """
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "dtypes": df.dtypes.value_counts().to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "columns": list(df.columns),
    }

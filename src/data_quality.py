"""
data_quality.py
===============
Módulo responsable de la **Fase 3**: auditoría de calidad del dataset.

Genera un reporte cuantitativo de:
- Valores faltantes (explícitos e implícitos como espacios en blanco).
- Duplicados de filas y de identificador único (`customerID`).
- Outliers en variables numéricas usando el método del rango intercuartílico (IQR).
- Rangos lógicos (p. ej. tenure no debería ser negativo).

Nota académica
--------------
En Telco Churn, los valores faltantes NO se presentan como NaN explícitos:
la columna `TotalCharges` contiene **strings vacíos (" ")** para 11 clientes
con `tenure == 0` (clientes nuevos que aún no han sido facturados).
Esta auditoría está diseñada para detectar ese tipo de problemas sutiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un reporte de valores faltantes contemplando dos tipos:
    1. NaN explícitos (detectados por pandas).
    2. Strings vacíos o solo-espacios en columnas tipo `object`.

    Returns
    -------
    pd.DataFrame con columnas: [variable, n_nan, n_blank, total_missing, pct_missing]
    """
    rows = []
    for col in df.columns:
        n_nan = df[col].isna().sum()
        # Para columnas tipo string (object en pandas <3, StringDtype en pandas 3):
        # contar strings que solo contienen espacios o vacíos.
        if pd.api.types.is_string_dtype(df[col]):
            n_blank = df[col].astype(str).str.strip().eq("").sum()
        else:
            n_blank = 0
        total = n_nan + n_blank
        rows.append({
            "variable": col,
            "n_nan": int(n_nan),
            "n_blank": int(n_blank),
            "total_missing": int(total),
            "pct_missing": round(100 * total / len(df), 3),
        })
    return pd.DataFrame(rows).sort_values("total_missing", ascending=False).reset_index(drop=True)


def duplicates_report(df: pd.DataFrame, id_col: str = "customerID") -> dict:
    """
    Reporta duplicados a nivel de fila completa y a nivel de ID único.
    """
    return {
        "duplicated_rows": int(df.duplicated().sum()),
        "duplicated_ids": int(df[id_col].duplicated().sum()),
        "unique_ids": int(df[id_col].nunique()),
    }


def outliers_iqr(series: pd.Series, k: float = 1.5) -> dict:
    """
    Detecta outliers usando el método del rango intercuartílico (IQR).

    Un valor es outlier si está fuera de [Q1 - k*IQR, Q3 + k*IQR].
    El factor `k=1.5` es el estándar de Tukey.

    Returns
    -------
    dict con número y porcentaje de outliers, y los límites usados.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (s < lower) | (s > upper)
    return {
        "n_outliers": int(mask.sum()),
        "pct_outliers": round(100 * mask.sum() / len(s), 3),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def numeric_outliers_report(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Aplica `outliers_iqr` a cada columna numérica indicada."""
    rows = []
    for col in numeric_cols:
        info = outliers_iqr(df[col])
        info["variable"] = col
        rows.append(info)
    return pd.DataFrame(rows).set_index("variable")


def logical_range_checks(df: pd.DataFrame) -> dict:
    """
    Verifica rangos lógicos específicos del dominio del problema.
    Si alguna regla se incumple, devuelve el conteo de violaciones.
    """
    checks = {
        "tenure_negativo": int((pd.to_numeric(df["tenure"], errors="coerce") < 0).sum()),
        "MonthlyCharges_negativo": int((pd.to_numeric(df["MonthlyCharges"], errors="coerce") < 0).sum()),
        "TotalCharges_no_numerico": int(
            pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()
            - df["TotalCharges"].isna().sum()
        ),
        "SeniorCitizen_fuera_01": int(~df["SeniorCitizen"].isin([0, 1])).sum() if df["SeniorCitizen"].dtype != bool else 0,
    }
    return checks

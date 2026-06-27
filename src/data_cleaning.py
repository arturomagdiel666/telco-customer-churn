"""
data_cleaning.py
================
Módulo responsable de la **Fase 4**: limpieza y preparación.

Aplica las decisiones documentadas tras la auditoría de calidad:

1. `TotalCharges`: convertir a numérico. Los 11 strings vacíos corresponden
   a clientes con `tenure == 0` (recién contratados, sin facturación todavía).
   Decisión: imputar con `0.0` porque tiene sentido de negocio (aún no han
   pagado nada). Alternativa: eliminarlos. Justificamos imputación porque
   son <0.2% y eliminarlos sesgaría la muestra de "nuevos clientes".

2. `SeniorCitizen`: viene como entero 0/1 pero semánticamente es categórico
   binario. Lo convertimos a "Yes"/"No" para homogeneidad con el resto de
   variables sí/no del dataset.

3. `Churn`: variable objetivo. Mapear "Yes"/"No" → 1/0 para el modelado.

4. Nombres de columnas: pasamos a `snake_case` (estándar PEP 8 para Python).

5. Creamos una variable derivada `tenure_group` (binning de la antigüedad
   en cuartiles temporales) que será útil para EDA y como feature adicional.
"""

from __future__ import annotations

import pandas as pd


def _to_snake_case(name: str) -> str:
    """Convierte 'MonthlyCharges' → 'monthly_charges'."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de limpieza al dataset crudo.

    Devuelve un nuevo DataFrame (no modifica el original) listo para EDA
    y para entrar a las fases de modelado.
    """
    df = df.copy()

    # --- 1) TotalCharges: string → float, imputar blancos con 0.0 ---
    # `errors="coerce"` convierte los strings vacíos en NaN, que luego
    # rellenamos con 0.0 (clientes con tenure=0 que aún no facturan).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # --- 2) SeniorCitizen: 0/1 → "No"/"Yes" para homogeneidad categórica ---
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # --- 3) Churn: "Yes"/"No" → 1/0 (variable objetivo numérica) ---
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # --- 4) Estandarizar nombres de columnas a snake_case ---
    df.columns = [_to_snake_case(c) for c in df.columns]

    # --- 5) Variable derivada: tenure_group (binning por años de antigüedad) ---
    # Justificación: el tenure (en meses) tiene relación no-lineal con churn.
    # Agruparlo en tramos facilita visualización y captura efectos por etapa.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )

    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """Guarda el dataset limpio en `data/processed/`."""
    df.to_csv(output_path, index=False)

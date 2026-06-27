"""
stats_tests.py
==============
Módulo responsable de las **Fases 6 y 7**: pruebas estadísticas y correlaciones.

Pruebas incluidas
-----------------
- **Shapiro-Wilk**: prueba de normalidad para variables numéricas
  (válido para n hasta ~5000; aquí muestreamos si excede).
- **Levene**: prueba de homogeneidad de varianzas entre grupos (churn vs no-churn).
- **Welch's t-test**: comparación de medias cuando no se asume igualdad de
  varianzas (más robusto que el t-test clásico).
- **Mann-Whitney U**: alternativa no paramétrica al t-test cuando se
  rechaza normalidad.
- **Chi-cuadrado de independencia**: asociación entre variables categóricas
  y la variable objetivo `churn`.
- **Cramér's V**: tamaño del efecto para la asociación chi-cuadrado
  (el p-valor solo dice "hay asociación", V dice "qué tan fuerte es").

Interpretación de p-valores
---------------------------
Usamos α = 0.05. Si p < 0.05, rechazamos H₀ (hay diferencia/asociación).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05


# ---------- Fase 6: pruebas estadísticas ----------

def normality_test(series: pd.Series, max_n: int = 5000, seed: int = 42) -> dict:
    """
    Shapiro-Wilk con muestreo si la serie excede `max_n`.
    Si rechaza H₀ (p < 0.05), la variable NO se distribuye normalmente.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    sampled = False
    if len(s) > max_n:
        s = s.sample(max_n, random_state=seed)
        sampled = True
    stat, p = stats.shapiro(s)
    return {
        "n_used": int(len(s)),
        "sampled": sampled,
        "shapiro_W": float(stat),
        "p_value": float(p),
        "is_normal": bool(p >= ALPHA),
    }


def levene_test(df: pd.DataFrame, num_col: str, group_col: str = "churn") -> dict:
    """Levene: ¿las varianzas son iguales entre grupos? (H₀: sí lo son)."""
    g0 = pd.to_numeric(df.loc[df[group_col] == 0, num_col], errors="coerce").dropna()
    g1 = pd.to_numeric(df.loc[df[group_col] == 1, num_col], errors="coerce").dropna()
    stat, p = stats.levene(g0, g1)
    return {
        "levene_stat": float(stat),
        "p_value": float(p),
        "equal_variance": bool(p >= ALPHA),
    }


def compare_groups(df: pd.DataFrame, num_col: str, group_col: str = "churn") -> dict:
    """
    Compara la distribución de `num_col` entre grupos definidos por `group_col`.
    Aplica:
        - Welch's t-test (no asume varianzas iguales).
        - Mann-Whitney U (alternativa no paramétrica).
    Reportar ambas permite decidir según la normalidad de la variable.
    """
    g0 = pd.to_numeric(df.loc[df[group_col] == 0, num_col], errors="coerce").dropna()
    g1 = pd.to_numeric(df.loc[df[group_col] == 1, num_col], errors="coerce").dropna()
    t_stat, t_p = stats.ttest_ind(g0, g1, equal_var=False)  # Welch
    u_stat, u_p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    return {
        "mean_group0": float(g0.mean()),
        "mean_group1": float(g1.mean()),
        "welch_t": float(t_stat),
        "welch_p": float(t_p),
        "mannwhitney_U": float(u_stat),
        "mannwhitney_p": float(u_p),
        "significant_welch": bool(t_p < ALPHA),
        "significant_mw": bool(u_p < ALPHA),
    }


def chi2_independence(df: pd.DataFrame, cat_col: str, target: str = "churn") -> dict:
    """
    Chi-cuadrado de independencia + Cramér's V (tamaño del efecto).

    Cramér's V interpretation (Cohen):
        ~0.1  → efecto pequeño
        ~0.3  → efecto moderado
        ~0.5  → efecto grande
    """
    table = pd.crosstab(df[cat_col], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else np.nan
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "significant": bool(p < ALPHA),
    }


def chi2_table(df: pd.DataFrame, cat_cols: list[str], target: str = "churn") -> pd.DataFrame:
    """Aplica chi² a cada columna categórica y devuelve tabla ordenada por Cramér's V."""
    rows = []
    for c in cat_cols:
        res = chi2_independence(df, c, target)
        res["variable"] = c
        rows.append(res)
    out = pd.DataFrame(rows).set_index("variable")
    return out.sort_values("cramers_v", ascending=False)


# ---------- Fase 7: correlaciones ----------

def correlation_with_target(df: pd.DataFrame, num_cols: list[str],
                             target: str = "churn", method: str = "spearman") -> pd.DataFrame:
    """
    Correlación de cada variable numérica con la variable objetivo.
    Usamos Spearman por defecto (no asume normalidad; capta relaciones monotónicas).
    """
    rows = []
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if method == "pearson":
            r, p = stats.pearsonr(s, df[target])
        else:
            r, p = stats.spearmanr(s, df[target])
        rows.append({"variable": col, "corr": float(r), "p_value": float(p)})
    out = pd.DataFrame(rows).set_index("variable")
    out["abs_corr"] = out["corr"].abs()
    return out.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")

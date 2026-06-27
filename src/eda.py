"""
eda.py
======
Módulo responsable de la **Fase 5**: Análisis Exploratorio de Datos (EDA).

Concentra funciones de visualización reutilizables para que el notebook
quede limpio y los gráficos sean consistentes en estilo.

Convenciones de estilo
----------------------
- Paleta: colores de la familia teal/coral para destacar churn vs no-churn.
- Tamaño base de figura: 8x5 (univariadas), 12x6 (correlación / heatmaps).
- Cada función devuelve la figura para que el notebook pueda mostrarla y/o
  guardarla en `reports/figures/`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paleta consistente: 0=no churn (teal), 1=churn (coral)
CHURN_PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
sns.set_theme(style="whitegrid", context="notebook")


def plot_target_distribution(df: pd.DataFrame, target: str = "churn") -> plt.Figure:
    """Distribución de la variable objetivo (porcentaje de churn)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df[target].value_counts().sort_index()
    pct = 100 * counts / counts.sum()
    bars = ax.bar(["No churn (0)", "Churn (1)"], counts.values,
                  color=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    for bar, p, c in zip(bars, pct.values, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{c:,}\n({p:.1f}%)", ha="center", fontsize=11)
    ax.set_title("Distribución de la variable objetivo (Churn)", fontsize=13)
    ax.set_ylabel("Número de clientes")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    """Histogramas + KDE para variables numéricas, separados por churn."""
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        for cls in [0, 1]:
            sns.histplot(df.loc[df["churn"] == cls, col],
                         ax=ax, kde=True, stat="density",
                         color=CHURN_PALETTE[cls], alpha=0.45,
                         label=f"Churn={cls}", bins=30)
        ax.set_title(f"Distribución de {col}")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_categorical_vs_churn(df: pd.DataFrame, cat_cols: list[str], ncols: int = 3) -> plt.Figure:
    """Tasa de churn por categoría para una lista de variables categóricas."""
    nrows = int(np.ceil(len(cat_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cat_cols):
        rate = df.groupby(col, observed=True)["churn"].mean().sort_values()
        ax.barh(rate.index.astype(str), rate.values * 100,
                color="#264653", edgecolor="white")
        ax.axvline(df["churn"].mean() * 100, color="#e76f51",
                   linestyle="--", linewidth=1, label="Tasa global")
        ax.set_xlabel("% Churn")
        ax.set_title(f"Tasa de churn por {col}")
        ax.legend(fontsize=8, loc="lower right")

    # Apagar ejes sobrantes
    for ax in axes[len(cat_cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    """Heatmap de correlación (Pearson) para variables numéricas."""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Matriz de correlación (Pearson) - variables numéricas")
    fig.tight_layout()
    return fig

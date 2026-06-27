"""
modeling.py
===========
Módulo de soporte para las **Fases 8 a 16**: selección de variables,
preprocesamiento, entrenamiento, validación cruzada, tuning, evaluación e
interpretación de los modelos de predicción de churn.

Filosofía del módulo
--------------------
El notebook ``02_modeling.ipynb`` debe leerse como un tutorial; por eso toda la
"fontanería" (construcción de pipelines, cálculo de métricas, gráficos) vive
aquí, en funciones puras, documentadas y testeables. El notebook solo orquesta
y narra.

Decisiones metodológicas clave (se justifican en cada función):

- **Pipeline de scikit-learn**: encadenamos preprocesamiento + modelo en un solo
  objeto. Así evitamos *data leakage* (el ``StandardScaler`` se ajusta solo con
  train dentro de cada fold de CV) y obtenemos un artefacto único, serializable,
  que el dashboard puede cargar y usar sobre datos crudos.

- **Métrica principal = F1 de la clase positiva (churn=1)**. El dataset está
  desbalanceado (~26.5% de churn). El *accuracy* premia predecir siempre "no
  churn"; F1 equilibra precisión y recall sobre la clase que de verdad importa.

- **Semilla global ``RANDOM_STATE = 42``** en todo lo aleatorio
  (split, K-Fold, búsquedas), para reproducibilidad total.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Semilla global: cualquier proceso aleatorio del módulo la usa.
RANDOM_STATE = 42

# Paleta consistente con el resto del proyecto (notebooks y dashboard).
CHURN_PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
MODEL_COLORS = {
    "Dummy": "#9aa0a6",
    "LogisticRegression": "#2a9d8f",
    "RandomForest": "#e9c46a",
    "XGBoost": "#e76f51",
}


# ----------------------------------------------------------------------------
# Fase 8 — Selección de variables
# ----------------------------------------------------------------------------

#: Variables que el EDA (Fases 6-7) descartó por NO mostrar asociación
#: significativa con churn (chi² no significativo). Las quitamos del modelado
#: para reducir ruido y dimensionalidad.
DROPPED_FEATURES = ["gender", "phone_service"]

#: Columnas que nunca son predictoras (identificador y target).
ID_COLS = ["customer_id"]
TARGET = "churn"


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """
    Separa una lista de features en numéricas y categóricas.

    Parameters
    ----------
    df : DataFrame con los datos.
    features : nombres de columnas a clasificar.

    Returns
    -------
    (numeric_cols, categorical_cols)

    Nota metodológica
    -----------------
    Nos basamos en el ``dtype`` real de la columna. ``senior_citizen`` viene
    como texto No/Yes, así que cae (correctamente) en categóricas.
    """
    numeric, categorical = [], []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def compute_vif(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Calcula el **VIF** (Variance Inflation Factor) de las variables numéricas.

    El VIF mide cuánta multicolinealidad aporta cada variable: VIF=1 → ninguna;
    VIF>5 → preocupante; VIF>10 → multicolinealidad severa. Sirve para decidir
    si dos variables casi redundantes (p. ej. ``tenure`` y ``total_charges``,
    r≈0.83) deben simplificarse.

    Implementación: VIF_i = 1 / (1 - R²_i), donde R²_i proviene de regredir la
    variable i contra todas las demás. Usamos ``statsmodels`` para el cálculo
    estándar.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
    X = add_constant(X)  # el VIF requiere intercepto
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append({"variable": col, "VIF": float(variance_inflation_factor(X.values, i))})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def mutual_information_ranking(
    df: pd.DataFrame, features: list[str], target: str = TARGET
) -> pd.DataFrame:
    """
    Ranking de relevancia por **información mutua** (MI) entre cada predictora y
    el target.

    A diferencia de Cramér's V (solo categóricas) o Spearman (solo monotónico),
    la MI captura dependencias de cualquier tipo, lineales o no, y funciona para
    variables mixtas. MI=0 → independencia; a mayor MI, más informativa.

    Las categóricas se codifican con códigos enteros y se marcan como discretas
    para que ``mutual_info_classif`` use el estimador adecuado.
    """
    X = pd.DataFrame(index=df.index)
    discrete_mask = []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            X[col] = pd.to_numeric(df[col], errors="coerce")
            discrete_mask.append(False)
        else:
            X[col] = df[col].astype("category").cat.codes
            discrete_mask.append(True)
    mi = mutual_info_classif(
        X.values, df[target].values, discrete_features=discrete_mask, random_state=RANDOM_STATE
    )
    return (
        pd.DataFrame({"variable": features, "mutual_info": mi})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------------------
# Fase 10 — Preprocesamiento (ColumnTransformer + Pipeline)
# ----------------------------------------------------------------------------

def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Construye el ``ColumnTransformer`` que estandariza numéricas y codifica
    categóricas.

    - **Numéricas → StandardScaler**: imprescindible para Logistic Regression
      (que es sensible a la escala y a la regularización); inocuo para
      Random Forest / XGBoost (que son invariantes a transformaciones
      monótonas), así que lo aplicamos a todos por simplicidad y un único
      pipeline homogéneo.
    - **Categóricas → OneHotEncoder(drop='first', handle_unknown='ignore')**:
      ``drop='first'`` evita la trampa de la variable dummy (colinealidad
      perfecta) que perjudica a modelos lineales; ``handle_unknown='ignore'``
      hace que el dashboard no se rompa si llega una categoría no vista.

    Devuelve un transformador sin ajustar; se ajusta dentro del Pipeline.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(preprocessor: ColumnTransformer, model: ClassifierMixin) -> Pipeline:
    """Encadena preprocesamiento + modelo en un único objeto serializable."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# ----------------------------------------------------------------------------
# Fase 11 / 13 — Catálogo de modelos y grillas de hiperparámetros
# ----------------------------------------------------------------------------

def scale_pos_weight(y: pd.Series) -> float:
    """
    Peso de la clase positiva para XGBoost = n_negativos / n_positivos.

    Es la forma canónica de comunicarle a XGBoost que la clase churn está en
    minoría, equivalente a ``class_weight='balanced'`` en LR/RF.
    """
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / pos if pos else 1.0


def get_base_models(y_train: pd.Series) -> dict[str, ClassifierMixin]:
    """
    Modelos con parámetros por defecto para el baseline (Fase 11).

    Incluye el **modelo trivial** (``DummyClassifier`` que predice la clase
    mayoritaria) como piso mínimo: cualquier modelo útil debe superarlo.
    """
    spw = scale_pos_weight(y_train)
    return {
        "Dummy": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=spw,
            n_jobs=-1,
        ),
    }


def get_param_distributions(y_train: pd.Series) -> dict[str, dict[str, list]]:
    """
    Grillas de búsqueda para ``RandomizedSearchCV`` (Fase 13).

    Las grillas se mantienen pequeñas (≤ ~20 combinaciones efectivas) para que
    la búsqueda termine en pocos minutos en una laptop. Los nombres de
    parámetros llevan el prefijo ``model__`` porque el estimador vive dentro de
    un Pipeline.
    """
    spw = scale_pos_weight(y_train)
    return {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
            # liblinear soporta tanto l1 como l2 y va bien en datasets medianos.
            "model__solver": ["liblinear"],
        },
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            # scale_pos_weight calculado a partir del desbalance real.
            "model__scale_pos_weight": [1.0, spw],
        },
    }


# ----------------------------------------------------------------------------
# Fases 11-14 — Evaluación
# ----------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    """
    Calcula el panel de métricas estándar para clasificación binaria.

    - accuracy: proporción global de aciertos (engañosa con desbalance).
    - precision (clase 1): de los marcados como churn, cuántos lo eran.
    - recall (clase 1): de los churners reales, cuántos detectamos.
    - f1 (clase 1): media armónica de las dos anteriores → métrica principal.
    - roc_auc / pr_auc: capacidad de ranking del modelo (requieren probabilidad).
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Ajusta (si hace falta) y evalúa un pipeline en train y test.

    Devuelve un dict con sub-dicts ``train`` y ``test`` de métricas. Comparar
    ambos revela **overfitting** (train mucho mejor que test).
    """
    pipeline.fit(X_train, y_train)
    res = {}
    for split, (X, y) in {"train": (X_train, y_train), "test": (X_test, y_test)}.items():
        y_pred = pipeline.predict(X)
        y_proba = (
            pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None
        )
        res[split] = compute_metrics(y.values, y_pred, y_proba)
    return res


def metrics_dataframe(results: dict[str, dict], split: str = "test") -> pd.DataFrame:
    """Convierte un dict {modelo: {train/test: métricas}} en tabla ordenable."""
    rows = {name: r[split] for name, r in results.items()}
    df = pd.DataFrame(rows).T
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    return df[[c for c in cols if c in df.columns]]


# ----------------------------------------------------------------------------
# Gráficos de evaluación
# ----------------------------------------------------------------------------

def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, normalize: bool = False
) -> plt.Figure:
    """Matriz de confusión (cruda o normalizada por fila) como heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    cm_display = cm
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm_display, cmap="BuGn")
    ax.set_xticks([0, 1], ["Pred: No churn", "Pred: Churn"])
    ax.set_yticks([0, 1], ["Real: No churn", "Real: Churn"])
    thresh = cm_display.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, format(cm_display[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black", fontsize=13,
            )
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    fitted: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> plt.Figure:
    """Curvas ROC superpuestas de varios modelos (excluye el Dummy)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, pipe in fitted.items():
        if name == "Dummy" or not hasattr(pipe, "predict_proba"):
            continue
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=MODEL_COLORS.get(name), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar (AUC=0.5)")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR / Recall)")
    ax.set_title("Curvas ROC — comparación de modelos")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curves(
    fitted: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> plt.Figure:
    """Curvas Precision-Recall superpuestas (más informativas con desbalance)."""
    baseline = float((y_test == 1).mean())
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, pipe in fitted.items():
        if name == "Dummy" or not hasattr(pipe, "predict_proba"):
            continue
        proba = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                color=MODEL_COLORS.get(name), linewidth=2)
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1,
               label=f"Azar (={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall — comparación de modelos")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Fase 15 — Importancia de variables e interpretación
# ----------------------------------------------------------------------------

def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Nombres de las features tras el preprocesamiento (num + one-hot)."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def get_feature_importance(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    """
    Importancia de variables del modelo ganador, homogeneizada a una tabla.

    - **RandomForest / XGBoost**: ``feature_importances_`` (ganancia / reducción
      de impureza). Siempre positivas → magnitud de relevancia.
    - **LogisticRegression**: coeficientes estandarizados. Como las features ya
      están escaladas, el valor absoluto del coeficiente es comparable entre
      variables; el signo indica la dirección (positivo → empuja a churn).
    """
    names = get_feature_names(pipeline)
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        df = pd.DataFrame({"feature": names, "importance": imp})
    elif hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        df = pd.DataFrame({"feature": names, "coef": coef, "importance": np.abs(coef)})
    else:
        raise ValueError(f"El modelo {model_name} no expone importancias.")
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(imp_df: pd.DataFrame, top_n: int = 15, title: str = "") -> plt.Figure:
    """Barras horizontales con las ``top_n`` variables más importantes."""
    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(top))))
    colors = "#264653"
    if "coef" in top.columns:
        colors = [CHURN_PALETTE[1] if c > 0 else CHURN_PALETTE[0] for c in top["coef"]]
    ax.barh(top["feature"], top["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Importancia" + (" (|coef| estandarizado)" if "coef" in top.columns else ""))
    ax.set_title(title or "Importancia de variables (top features)")
    fig.tight_layout()
    return fig


def error_profile(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    kind: str = "fn",
) -> pd.DataFrame:
    """
    Devuelve el subconjunto de test correspondiente a un tipo de error.

    - ``fn`` (falso negativo): churner real que el modelo NO detectó (y=1, pred=0).
      Son los casos más costosos: clientes que se van sin alerta.
    - ``fp`` (falso positivo): cliente fiel marcado como churn (y=0, pred=1).
      Generan gasto de retención innecesario.
    """
    y_test = y_test.reset_index(drop=True)
    X = X_test.reset_index(drop=True)
    if kind == "fn":
        mask = (y_test == 1) & (y_pred == 0)
    elif kind == "fp":
        mask = (y_test == 0) & (y_pred == 1)
    else:
        raise ValueError("kind debe ser 'fn' o 'fp'")
    return X.loc[mask.values]

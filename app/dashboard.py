"""
dashboard.py
============
Dashboard **Streamlit multipágina** del proyecto Customer Churn (Fase 17).

Es el front-end interactivo que consume los artefactos generados por los
notebooks 01 y 02:

- ``data/processed/churn_clean.csv``  → datos limpios para exploración.
- ``models/best_model.pkl``           → pipeline ganador (RandomForest).
- ``models/metadata.json``            → features y rangos válidos por feature.
- ``models/model_comparison.json``    → métricas de los 3 modelos + curvas.

Diseño didáctico
----------------
Cada página explica **qué** muestra y **cómo** leerlo, para que un estudiante la
navegue sin documentación externa. Paleta consistente con los notebooks:
teal #2a9d8f (no-churn) / coral #e76f51 (churn).

Ejecutar desde la raíz del proyecto::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Resolución de rutas robusta (independiente del cwd) ---------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "churn_clean.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
META_PATH = ROOT / "models" / "metadata.json"
COMPARISON_PATH = ROOT / "models" / "model_comparison.json"
FIG_DIR = ROOT / "reports" / "figures"

from src import modeling as M          # noqa: E402  (tras ajustar sys.path)
from src import stats_tests as ST      # noqa: E402

# --- Paleta y configuración global -------------------------------------------
TEAL = "#2a9d8f"      # no-churn
CORAL = "#e76f51"     # churn
NAVY = "#264653"
CHURN_COLORS = {0: TEAL, 1: CORAL}

st.set_page_config(page_title="Customer Churn · Dashboard", page_icon="📉", layout="wide")


# ============================================================================
# Cargas cacheadas
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Carga el dataset limpio. Cacheado: se lee del disco una sola vez."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    """Carga el pipeline serializado (preprocesamiento + modelo) con joblib."""
    import joblib
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_comparison() -> dict:
    with open(COMPARISON_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    """TreeExplainer del modelo ganador (RF). Cacheado por ser costoso de crear."""
    import shap
    return shap.TreeExplainer(_model.named_steps["model"])


def require_model() -> bool:
    """Verifica que exista el modelo entrenado; si no, instruye al usuario."""
    if not MODEL_PATH.exists():
        st.error(
            "⚠️ No se encontró `models/best_model.pkl`.\n\n"
            "Antes de usar el dashboard debes ejecutar el notebook de modelado:\n\n"
            "```\njupyter notebook notebooks/02_modeling.ipynb  # Run All\n```\n\n"
            "Ese notebook genera el modelo, la metadata y la comparación que esta "
            "página necesita."
        )
        return False
    return True


def footer() -> None:
    """Pie de página académico, idéntico en todas las páginas."""
    st.markdown("---")
    st.caption("📉 Customer Churn · Material académico — Curso Python Clase 9")


# ============================================================================
# PÁGINA 1 — Inicio
# ============================================================================
def page_inicio(df: pd.DataFrame) -> None:
    st.title("📉 Customer Churn")
    st.subheader("Análisis predictivo de fuga de clientes · Telco Customer Churn")

    # KPIs principales
    total = len(df)
    churn_rate = df["churn"].mean()
    median_tenure = pd.to_numeric(df["tenure"]).median()
    avg_monthly = pd.to_numeric(df["monthly_charges"]).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Clientes", f"{total:,}")
    c2.metric("🔥 Tasa de churn", f"{churn_rate*100:.1f}%")
    c3.metric("📅 Antigüedad mediana", f"{median_tenure:.0f} meses")
    c4.metric("💵 Cargo mensual medio", f"${avg_monthly:.1f}")

    st.markdown(
        """
        ### El problema
        Una empresa de telecomunicaciones quiere **anticipar qué clientes están a
        punto de irse** (*churn*). Retener es entre 5 y 25 veces más barato que
        adquirir, así que predecir la fuga permite actuar **antes** con campañas
        de retención dirigidas.

        Este dashboard responde tres preguntas:
        1. **¿Qué tipo de clientes** tienen más probabilidad de irse?
        2. **¿El contrato** mensual vs anual afecta la retención?
        3. **¿Qué recomendaciones** reducen la fuga?
        """
    )

    st.info(
        "**🧭 Cómo navegar este dashboard** (menú lateral izquierdo):\n\n"
        "- **📊 Exploración** — filtra y visualiza los datos crudos.\n"
        "- **🧪 Pruebas estadísticas** — evidencia formal de qué variables importan.\n"
        "- **🤖 Modelado** — comparación de los 3 modelos y por qué ganó RandomForest.\n"
        "- **🔮 Predicción** — calcula el riesgo de churn de un cliente concreto.\n"
        "- **📑 Reporte Científico** — conclusiones, recomendaciones y video del proyecto."
    )
    footer()


# ============================================================================
# PÁGINA 2 — Exploración de datos
# ============================================================================
def page_exploracion(df: pd.DataFrame) -> None:
    st.title("📊 Exploración de datos")
    st.write(
        "Explora el dataset con filtros interactivos. Las visualizaciones usan "
        "**Plotly**: pasa el cursor por encima para ver detalles, y usa la leyenda "
        "para mostrar/ocultar series."
    )

    # --- Filtros laterales ---
    st.sidebar.markdown("### 🔎 Filtros (Exploración)")
    f_contract = st.sidebar.multiselect("Contrato", sorted(df["contract"].unique()),
                                        default=sorted(df["contract"].unique()))
    f_internet = st.sidebar.multiselect("Servicio de internet",
                                        sorted(df["internet_service"].unique()),
                                        default=sorted(df["internet_service"].unique()))
    f_senior = st.sidebar.multiselect("Senior citizen", sorted(df["senior_citizen"].unique()),
                                      default=sorted(df["senior_citizen"].unique()))
    f_tenure = st.sidebar.multiselect("Grupo de antigüedad",
                                      sorted(df["tenure_group"].unique()),
                                      default=sorted(df["tenure_group"].unique()))

    mask = (
        df["contract"].isin(f_contract)
        & df["internet_service"].isin(f_internet)
        & df["senior_citizen"].isin(f_senior)
        & df["tenure_group"].isin(f_tenure)
    )
    fdf = df[mask]

    st.markdown(f"**Clientes seleccionados:** {len(fdf):,} de {len(df):,} "
                f"· tasa de churn en la selección: **{fdf['churn'].mean()*100:.1f}%**"
                if len(fdf) else "**Sin clientes** para los filtros elegidos.")

    st.dataframe(fdf, use_container_width=True, height=280)
    st.download_button(
        "⬇️ Descargar dataset filtrado (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="churn_filtrado.csv",
        mime="text/csv",
    )

    if fdf.empty:
        footer()
        return

    st.markdown("### Distribuciones")
    col1, col2 = st.columns(2)

    # Distribución del target
    counts = fdf["churn"].map({0: "No churn", 1: "Churn"}).value_counts()
    fig_t = px.bar(x=counts.index, y=counts.values,
                   color=counts.index, color_discrete_map={"No churn": TEAL, "Churn": CORAL},
                   labels={"x": "", "y": "Clientes"}, title="Distribución del target")
    fig_t.update_layout(showlegend=False)
    col1.plotly_chart(fig_t, use_container_width=True)

    # Heatmap de correlación numérica
    num_cols = ["tenure", "monthly_charges", "total_charges"]
    corr = fdf[num_cols].apply(pd.to_numeric).corr()
    fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, title="Correlación entre variables numéricas")
    col2.plotly_chart(fig_c, use_container_width=True)

    # Histogramas segmentados por churn
    st.markdown("#### Histogramas por estado de churn")
    var = st.selectbox("Variable numérica", num_cols, index=0)
    hdf = fdf.copy()
    hdf["Estado"] = hdf["churn"].map({0: "No churn", 1: "Churn"})
    fig_h = px.histogram(hdf, x=var, color="Estado", barmode="overlay", nbins=40,
                         color_discrete_map={"No churn": TEAL, "Churn": CORAL},
                         marginal="box", title=f"Distribución de {var} por churn")
    st.plotly_chart(fig_h, use_container_width=True)

    # Tasa de churn por categoría (selector)
    st.markdown("#### Tasa de churn por categoría")
    cat_options = ["contract", "internet_service", "payment_method", "tech_support",
                   "online_security", "senior_citizen", "paperless_billing", "tenure_group"]
    cat = st.selectbox("Variable categórica", cat_options, index=0)
    rate = (fdf.groupby(cat, observed=True)["churn"].mean() * 100).sort_values()
    fig_r = px.bar(x=rate.values, y=rate.index.astype(str), orientation="h",
                   labels={"x": "% Churn", "y": cat}, title=f"Tasa de churn por {cat}",
                   color=rate.values, color_continuous_scale=["#2a9d8f", "#e9c46a", "#e76f51"])
    fig_r.add_vline(x=fdf["churn"].mean()*100, line_dash="dash", line_color=NAVY,
                    annotation_text="Tasa global")
    fig_r.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_r, use_container_width=True)

    footer()


# ============================================================================
# PÁGINA 3 — Pruebas estadísticas
# ============================================================================
@st.cache_data(show_spinner=False)
def compute_numeric_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Shapiro / Levene / Welch / Mann-Whitney para cada variable numérica."""
    rows = []
    for col in ["tenure", "monthly_charges", "total_charges"]:
        norm = ST.normality_test(df[col])
        lev = ST.levene_test(df, col)
        cmp = ST.compare_groups(df, col)
        rows.append({
            "variable": col,
            "Shapiro p": norm["p_value"], "¿Normal?": "No" if not norm["is_normal"] else "Sí",
            "Levene p": lev["p_value"], "¿Var. iguales?": "Sí" if lev["equal_variance"] else "No",
            "Welch p": cmp["welch_p"], "Mann-Whitney p": cmp["mannwhitney_p"],
            "media (no churn)": cmp["mean_group0"], "media (churn)": cmp["mean_group1"],
        })
    return pd.DataFrame(rows).set_index("variable")


@st.cache_data(show_spinner=False)
def compute_chi2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Chi² + Cramér's V para todas las variables categóricas (incl. descartadas)."""
    # pandas 3.0 carga las columnas de texto como dtype 'str' (no 'object'); por eso
    # seleccionamos las NO numéricas en vez de filtrar por ==object.
    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and c not in ("customer_id", "churn")]
    tbl = ST.chi2_table(df, cat_cols)
    tbl = tbl.rename(columns={"chi2": "Chi²", "p_value": "p-valor",
                              "cramers_v": "Cramér's V", "significant": "¿Significativa?"})
    tbl["¿Significativa?"] = tbl["¿Significativa?"].map({True: "✅ Sí", False: "❌ No"})
    return tbl[["Chi²", "p-valor", "Cramér's V", "dof", "¿Significativa?"]]


def page_estadistica(df: pd.DataFrame) -> None:
    st.title("🧪 Pruebas estadísticas")
    st.write(
        "Evidencia formal de **qué variables se asocian con el churn**, calculada "
        "en vivo sobre el dataset limpio (`src/stats_tests.py`)."
    )

    st.markdown("### Variables numéricas")
    # 'variable' como columna (no índice) para que no se recorten los nombres largos.
    num_tbl = compute_numeric_tests(df).reset_index()
    st.dataframe(num_tbl.style.format({
        "Shapiro p": "{:.1e}", "Levene p": "{:.1e}", "Welch p": "{:.1e}",
        "Mann-Whitney p": "{:.1e}", "media (no churn)": "{:.1f}", "media (churn)": "{:.1f}",
    }), use_container_width=True, hide_index=True)
    st.caption(
        "📖 **Cómo leerlo:** *Shapiro p < 0.05* → la variable **no** es normal (por eso "
        "usamos Mann-Whitney además de Welch). *Welch/Mann-Whitney p < 0.05* → la media/"
        "distribución **difiere** entre clientes que se van y que se quedan: la variable "
        "discrimina churn."
    )

    st.markdown("### Variables categóricas — Chi² + Cramér's V")
    chi_tbl = compute_chi2_table(df)
    # Altura suficiente para mostrar TODAS las categóricas sin scroll interno
    # (incluye gender / phone_service, las no significativas que cita el pie).
    chi_height = (len(chi_tbl) + 1) * 35 + 3
    st.dataframe(chi_tbl.style.format({"Chi²": "{:.1f}", "p-valor": "{:.1e}",
                                       "Cramér's V": "{:.3f}"})
                 .background_gradient(subset=["Cramér's V"], cmap="YlOrRd"),
                 use_container_width=True, height=chi_height)
    st.caption(
        "📖 **Cómo leerlo:** el *p-valor* dice **si** hay asociación (p < 0.05 → sí); "
        "*Cramér's V* dice **cuán fuerte** es (≈0.1 débil, ≈0.3 moderada, ≈0.5 fuerte). "
        "`gender` y `phone_service` salen **no significativas** → se descartaron del modelo."
    )
    footer()


# ============================================================================
# PÁGINA 4 — Modelado y comparación
# ============================================================================
def page_modelado(comp: dict) -> None:
    st.title("🤖 Modelado y comparación")
    st.write(
        "Comparación de los tres algoritmos sobre el **set de test** (20% no visto). "
        "Métrica principal: **F1 de la clase churn**."
    )

    order = ["LogisticRegression", "RandomForest", "XGBoost"]
    label_map = {"f1": "F1 (churn)", "recall": "Recall", "precision": "Precision",
                 "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "accuracy": "Accuracy"}
    metrics_df = pd.DataFrame({m: comp["metrics"][m] for m in order}).T
    metrics_df = metrics_df[["f1", "recall", "precision", "pr_auc", "roc_auc", "accuracy"]]
    metrics_df = metrics_df.rename(columns=label_map)
    winner = comp["winner"]
    metrics_df.index = [f"🏆 {i}" if i == winner else i for i in metrics_df.index]

    st.markdown("### Tabla comparativa (test) — mejor valor por métrica resaltado")
    # El nombre del modelo va en una columna propia (no en el índice) para que
    # Streamlit le dé ancho suficiente y no recorte "🏆 RandomForest".
    metric_cols = list(metrics_df.columns)
    disp = metrics_df.reset_index(names="Modelo")
    st.dataframe(
        disp.style.format({c: "{:.4f}" for c in metric_cols})
            .highlight_max(subset=metric_cols, axis=0, color="#9fe2bf"),
        use_container_width=True, hide_index=True,
    )

    # --- Nota metodológica obligatoria (empate técnico RF vs XGBoost) ---
    st.warning(f"**📌 Nota metodológica.** {comp['methodological_note']}")

    # Barras comparativas
    melt = metrics_df.reset_index().melt(id_vars="index", var_name="Métrica", value_name="Valor")
    fig_bar = px.bar(melt, x="Métrica", y="Valor", color="index", barmode="group",
                     title="Métricas por modelo",
                     color_discrete_map={f"🏆 {winner}": CORAL,
                                         "LogisticRegression": TEAL, "XGBoost": "#e9c46a"})
    fig_bar.update_layout(legend_title="", yaxis_range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

    # Curvas ROC y PR superpuestas
    c1, c2 = st.columns(2)
    color_map = {"LogisticRegression": TEAL, "RandomForest": CORAL, "XGBoost": "#e9c46a"}
    fig_roc = go.Figure()
    for name in order:
        r = comp["roc"][name]
        fig_roc.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                     name=f"{name} (AUC={r['auc']:.3f})",
                                     line=dict(color=color_map[name])))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                                 line=dict(dash="dash", color="gray")))
    fig_roc.update_layout(title="Curvas ROC", xaxis_title="FPR", yaxis_title="TPR",
                          legend=dict(x=0.4, y=0.1))
    c1.plotly_chart(fig_roc, use_container_width=True)

    fig_pr = go.Figure()
    for name in order:
        p = comp["pr"][name]
        fig_pr.add_trace(go.Scatter(x=p["recall"], y=p["precision"], mode="lines",
                                    name=f"{name} (AP={p['ap']:.3f})",
                                    line=dict(color=color_map[name])))
    fig_pr.add_hline(y=comp["churn_rate_test"], line_dash="dash", line_color="gray",
                     annotation_text="Azar")
    fig_pr.update_layout(title="Curvas Precision-Recall", xaxis_title="Recall",
                         yaxis_title="Precision", legend=dict(x=0.4, y=0.95))
    c2.plotly_chart(fig_pr, use_container_width=True)

    # Matriz de confusión del ganador + feature importance
    c3, c4 = st.columns(2)
    cm = np.array(comp["confusion_winner"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="BuGn",
                       x=["Pred: No churn", "Pred: Churn"],
                       y=["Real: No churn", "Real: Churn"],
                       title=f"Matriz de confusión — {winner} (test)")
    fig_cm.update_layout(coloraxis_showscale=False)
    c3.plotly_chart(fig_cm, use_container_width=True)

    imp = pd.DataFrame(comp["feature_importance"]).sort_values("importance")
    fig_imp = px.bar(imp, x="importance", y="feature", orientation="h",
                     title=f"Importancia de variables (top 15) — {winner}",
                     color="importance", color_continuous_scale=["#2a9d8f", "#e76f51"])
    fig_imp.update_layout(coloraxis_showscale=False, height=480)
    c4.plotly_chart(fig_imp, use_container_width=True)

    footer()


# ============================================================================
# PÁGINA 5 — Predicción interactiva
# ============================================================================
def risk_band(p: float) -> tuple[str, str]:
    """Devuelve (etiqueta, función-streamlit) según la probabilidad."""
    if p < 0.30:
        return "🟢 Riesgo BAJO", "success"
    if p < 0.60:
        return "🟡 Riesgo MEDIO", "warning"
    return "🔴 Riesgo ALTO", "error"


def page_prediccion(df: pd.DataFrame, model, meta: dict) -> None:
    st.title("🔮 Predicción interactiva de churn")
    st.write(
        "Introduce el perfil de un cliente y el modelo (**RandomForest**) estimará su "
        "probabilidad de fuga. Los rangos provienen de `models/metadata.json`."
    )

    with st.form("cliente"):
        inputs = {}
        cols = st.columns(3)
        # Numéricas → sliders
        for i, col in enumerate(meta["numeric_features"]):
            r = meta["numeric_ranges"][col]
            inputs[col] = cols[i % 3].slider(
                col, float(r["min"]), float(r["max"]), float(r["median"]),
            )
        # Categóricas → selectboxes
        cat_feats = meta["categorical_features"]
        for j, col in enumerate(cat_feats):
            opts = meta["categorical_values"][col]
            inputs[col] = cols[j % 3].selectbox(col, opts)
        submitted = st.form_submit_button("⚡ Predecir riesgo de churn")

    if not submitted:
        st.caption("Completa el formulario y pulsa **Predecir**.")
        footer()
        return

    X = pd.DataFrame([inputs])[meta["feature_order"]]
    proba = float(model.predict_proba(X)[0, 1])

    c1, c2 = st.columns([1, 1])
    # Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=proba * 100,
        number={"suffix": "%"},
        title={"text": "Probabilidad de churn"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": NAVY},
            "steps": [
                {"range": [0, 30], "color": "#cdeee6"},
                {"range": [30, 60], "color": "#fbe7c6"},
                {"range": [60, 100], "color": "#f6c4b6"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": proba * 100},
        },
    ))
    gauge.update_layout(height=300, margin=dict(t=50, b=10))
    c1.plotly_chart(gauge, use_container_width=True)

    label, kind = risk_band(proba)
    with c2:
        st.markdown("### Resultado")
        getattr(st, kind)(f"## {label}\n\n**Probabilidad de churn: {proba*100:.1f}%**")
        st.caption("Bandas: 🟢 < 30% · 🟡 30–60% · 🔴 > 60%")

    # --- Explicación SHAP del cliente concreto ---
    st.markdown("### ¿Por qué? Explicación SHAP de esta predicción")
    st.caption("Las barras rojas empujan hacia **churn**; las azules hacia **retención**.")
    try:
        import shap
        import matplotlib.pyplot as plt
        explainer = get_shap_explainer(model)
        pre = model.named_steps["preprocessor"]
        feat_names = M.get_feature_names(model)
        Xt = pre.transform(X)
        expl = explainer(Xt)
        vals, base = expl.values, expl.base_values
        if getattr(vals, "ndim", 2) == 3:      # clasificador binario con eje de clase
            vals, base = vals[:, :, 1], base[:, 1]
        ex = shap.Explanation(
            values=vals[0],
            base_values=float(base[0]) if np.ndim(base) else float(base),
            data=Xt[0], feature_names=feat_names,
        )
        shap.plots.waterfall(ex, max_display=12, show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:  # SHAP es opcional; no debe romper la predicción
        st.info(f"No se pudo generar la explicación SHAP ({exc}).")

    footer()


# ============================================================================
# PÁGINA 6 — Reporte Científico
# ============================================================================
def page_reporte(df: pd.DataFrame, comp: dict, meta: dict) -> None:
    st.title("📑 Reporte Científico")

    m = meta["metrics_test"]
    st.markdown("## 6.1 Resumen ejecutivo")
    st.markdown(
        f"""
        El proyecto analiza la **fuga de clientes** de una telco (7,043 clientes,
        ~26.5% de churn) para anticipar quién se irá y por qué. Tras un EDA y
        pruebas estadísticas (notebook 01), se entrenaron y compararon tres modelos
        (notebook 02). El ganador, **{comp['winner']}**, alcanza
        **F1 = {m['f1']:.3f}**, **recall = {m['recall']:.3f}** y
        **ROC-AUC = {m['roc_auc']:.3f}** en test: identifica a ~{m['recall']*100:.0f}%
        de los churners, permitiendo **priorizar campañas de retención** sobre los
        clientes de mayor riesgo en lugar de actuar a ciegas.

        Los principales motores de fuga son el **tipo de contrato**, la **antigüedad**
        del cliente y la **ausencia de servicios de soporte**. Las recomendaciones
        (§6.3) apuntan a migrar clientes a contratos anuales, reforzar el *onboarding*
        de los primeros meses y promover paquetes de seguridad/soporte.
        """
    )

    st.markdown("## 6.2 Hipótesis y resultados")
    hyp = pd.DataFrame([
        ["H1", "El tipo de contrato influye en el churn", "✅ Validada",
         "`contract` es la variable #1 (Cramér's V=0.41, MI e importancia). Two year = mayor protección."],
        ["H2", "A menor antigüedad (tenure), mayor churn", "✅ Validada",
         "`tenure` es la numérica más importante; los clientes 0–12 m concentran la fuga."],
        ["H3", "La falta de servicios de soporte aumenta el churn", "✅ Validada",
         "`online_security` y `tech_support` figuran en el top de MI e importancia."],
    ], columns=["Hipótesis", "Enunciado", "Estado", "Evidencia"]).set_index("Hipótesis")
    st.table(hyp)

    st.markdown("## 6.3 Respuesta a las 3 preguntas del proyecto")
    # Cifras de apoyo
    churn_m2m = df.loc[df["contract"] == "Month-to-month", "churn"].mean()
    churn_two = df.loc[df["contract"] == "Two year", "churn"].mean()
    churn_new = df.loc[df["tenure_group"] == "0-12m", "churn"].mean()

    st.markdown(
        f"""
        **1) ¿Qué tipo de clientes tienen más probabilidad de irse?**
        Perfil de alto riesgo: contrato **mes-a-mes**, **antigüedad baja** (0–12
        meses, churn ≈ {churn_new*100:.0f}%), **fibra óptica**, **sin** seguridad
        online ni soporte técnico, y pago por **electronic check**.

        **2) ¿El contrato mensual vs anual afecta la retención?**
        Drásticamente. El churn pasa de **≈ {churn_m2m*100:.0f}%** en contratos
        **mes-a-mes** a **≈ {churn_two*100:.0f}%** en contratos **a dos años**:
        una diferencia de ~{(churn_m2m-churn_two)*100:.0f} puntos. El contrato es el
        factor protector más fuerte del modelo.

        **3) Recomendaciones para reducir la fuga (accionables):**
        - **Incentivar contratos anuales** (descuento/beneficio por migrar desde
          mes-a-mes): es la palanca de mayor impacto esperado.
        - **Reforzar el *onboarding*** en los primeros 12 meses (seguimiento proactivo).
        - **Empaquetar seguridad online y soporte técnico**, asociados a menor churn.
        - **Revisar la experiencia de *fibra óptica***, segmento con churn elevado.
        - **Promover métodos de pago automáticos** frente a *electronic check*.
        - **Usar el modelo para priorizar**: enfocar retención en el top de riesgo
          (recall ~{m['recall']*100:.0f}%) en lugar de campañas masivas.
        - **Medir con A/B testing** el efecto real de cada acción (§6.5).
        """
    )

    st.markdown("## 6.4 Limitaciones del estudio")
    st.markdown(
        """
        - **Datos transversales (snapshot):** sin secuencia temporal no se modela
          *cuándo* ocurrirá la fuga, solo la propensión actual.
        - **No causalidad:** las asociaciones (p. ej. fibra ↔ churn) no implican
          causa; podría haber variables confusoras (precio, zona, competencia).
        - **Variables ausentes:** no hay quejas, tickets de soporte, NPS ni datos de
          competencia, que probablemente mejorarían la predicción de los casos difíciles.
        - **Imputación de `TotalCharges`:** 11 clientes con `tenure=0` se imputaron
          con 0.0 (< 0.2% del dataset); razonable pero introduce un supuesto.
        """
    )
    st.warning(f"**📌 Nota metodológica (honestidad científica).** {comp['methodological_note']}")

    st.markdown("## 6.5 Próximos pasos")
    st.markdown(
        """
        - **A/B testing** de las recomendaciones para estimar impacto causal real.
        - **Monitoreo en producción** del *drift* de datos y del rendimiento del modelo.
        - **Reentrenamiento periódico** con datos frescos y nuevas variables (quejas, NPS).
        - **Validar el empate RF/XGBoost** con bootstrap o un segundo split temporal.
        """
    )

    footer()


# ============================================================================
# PÁGINA 7 — Video Explicativo
# ============================================================================
def page_video() -> None:
    st.title("🎬 Video Explicativo del Proyecto")
    st.write(
        "Como cierre del proyecto, te invitamos a ver este breve video que resume "
        "el problema, el análisis y los resultados principales. ▶️"
    )

    # Ruta ASCII (sin acentos) para evitar problemas de encoding Windows→Linux.
    video_path = ROOT / "video" / "prediccion_fuga_clientes.mp4"
    if video_path.exists():
        st.video(str(video_path))
    else:
        st.warning("El archivo de video no está disponible en este momento.")

    st.caption(
        "⚠️ El contenido de este video fue generado con inteligencia artificial "
        "(NotebookLM) a partir del reporte académico del proyecto."
    )
    footer()


# ============================================================================
# Router principal
# ============================================================================
def main() -> None:
    st.sidebar.title("📉 Customer Churn")
    st.sidebar.caption("Dashboard académico · Clase 9")
    page = st.sidebar.radio(
        "Navegación",
        ["🏠 Inicio", "📊 Exploración de datos", "🧪 Pruebas estadísticas",
         "🤖 Modelado y comparación", "🔮 Predicción interactiva", "📑 Reporte Científico",
         "🎬 Video Explicativo"],
    )

    if not require_model():
        st.stop()

    df = load_data()
    model = load_model()
    meta = load_metadata()
    comp = load_comparison()

    if page == "🏠 Inicio":
        page_inicio(df)
    elif page == "📊 Exploración de datos":
        page_exploracion(df)
    elif page == "🧪 Pruebas estadísticas":
        page_estadistica(df)
    elif page == "🤖 Modelado y comparación":
        page_modelado(comp)
    elif page == "🔮 Predicción interactiva":
        page_prediccion(df, model, meta)
    elif page == "📑 Reporte Científico":
        page_reporte(df, comp, meta)
    elif page == "🎬 Video Explicativo":
        page_video()


if __name__ == "__main__":
    main()

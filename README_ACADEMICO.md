# Customer Churn – Proyecto de Ciencia de Datos

> **Propósito académico.** Análisis end-to-end de fuga de clientes (churn) sobre
> el dataset *Telco Customer Churn* de IBM (Kaggle).

## Preguntas que responde el proyecto

1. ¿Qué tipo de clientes tienen más probabilidad de irse?
2. ¿El contrato mensual vs anual afecta la retención?
3. ¿Qué recomendaciones se pueden dar para reducir la fuga?

## Estructura del proyecto

```
customer_churn_project/
├── data/
│   ├── raw/CustomerChurn.csv        ← dataset original
│   └── processed/churn_clean.csv    ← dataset limpio (generado)
├── notebooks/
│   ├── 01_etl_eda.ipynb             ← Fases 1-7 (ETL, EDA, estadística)
│   └── 02_modeling.ipynb            ← Fases 8-16 (modelado, CV, evaluación, SHAP)
├── src/                             ← módulos reutilizables y testeables
│   ├── data_loader.py               ← Fase 2 (carga + validación de esquema)
│   ├── data_quality.py              ← Fase 3 (auditoría de calidad)
│   ├── data_cleaning.py             ← Fase 4 (ETL)
│   ├── eda.py                       ← Fase 5 (visualización)
│   ├── stats_tests.py               ← Fases 6-7 (pruebas estadísticas y correlación)
│   └── modeling.py                  ← Fases 8-16 (pipeline, entrenamiento, comparación)
├── models/                          ← artefactos serializados
│   ├── best_model.pkl               ← pipeline ganador (RandomForest)
│   ├── metadata.json                ← features, rangos, hiperparámetros, métricas
│   └── model_comparison.json        ← métricas 3 modelos, curvas, nota metodológica
├── app/
│   └── dashboard.py                 ← Fase 17 (dashboard Streamlit, 6 páginas)
├── reports/
│   ├── report.md / report.pdf       ← Fases 18-20 (reporte académico final)
│   ├── build_pdf.py                 ← conversor report.md → report.pdf (Chromium)
│   └── figures/                     ← gráficos PNG (+ figures/dashboard/ capturas reales)
├── requirements.txt
└── README.md
```

## Estado del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Definición del problema | ✅ |
| 2 | Carga del dataset | ✅ |
| 3 | Auditoría de calidad | ✅ |
| 4 | Limpieza y ETL | ✅ |
| 5 | EDA | ✅ |
| 6 | Pruebas estadísticas | ✅ |
| 7 | Correlaciones | ✅ |
| 8 | Selección de variables | ✅ |
| 9 | División train/test | ✅ |
| 10 | Escalado y encoding | ✅ |
| 11–14 | Modelado, CV, tuning, evaluación | ✅ |
| 15 | Análisis de errores | ✅ |
| 16 | Interpretación (SHAP) | ✅ |
| 17 | Dashboard web | ✅ |
| 18 | Reporte final (`reports/report.md` + `.pdf`) | ✅ |
| 19 | Documentación / README | ✅ |
| 20 | Reproducibilidad y cierre | ✅ |

**🎉 Proyecto completo (Fases 1–20).**

## Cómo ejecutar

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar los notebooks (en orden)

```bash
jupyter notebook notebooks/01_etl_eda.ipynb     # ETL + EDA + estadística
jupyter notebook notebooks/02_modeling.ipynb    # modelado + artefactos en models/
```

Ejecuta las celdas en orden (Run All). Los notebooks están pensados para leerse
como un tutorial: cada sección explica **qué hace** y **por qué**. El notebook 02
genera `models/best_model.pkl`, `models/metadata.json` y `models/model_comparison.json`,
que el dashboard y el reporte consumen.

## Cómo correr el dashboard

Dashboard interactivo de 6 páginas en **Streamlit** (Inicio, Exploración, Pruebas
estadísticas, Modelado, Predicción interactiva y Reporte científico):

```bash
streamlit run app/dashboard.py
```

Se abre en `http://localhost:8501`. **Requisito previo:** haber ejecutado el
notebook `02_modeling.ipynb` (genera los artefactos en `models/`). La página
**Predicción** permite introducir el perfil de un cliente y obtener su probabilidad
de churn con un *gauge* de riesgo y una explicación **SHAP** por cliente.

## Resultados finales

Comparación de los tres modelos sobre el conjunto de **test (20 % no visto, split estratificado)**:

| Modelo | F1 (churn) | Recall | Precision | PR-AUC | ROC-AUC | Accuracy |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.6144 | 0.7861 | 0.5043 | 0.6338 | 0.8421 | 0.7381 |
| **🏆 RandomForest** | **0.6306** | 0.7781 | **0.5301** | 0.6574 | **0.8426** | **0.7580** |
| XGBoost | 0.6204 | **0.7888** | 0.5113 | **0.6640** | 0.8414 | 0.7438 |

**Modelo ganador: RandomForest** (`n_estimators=400`, `max_depth=10`,
`min_samples_split=10`, `class_weight='balanced'`). Mejor F1 en CV = 0.637,
consistente con el F1 de test (0.631) → sin sobreajuste.

> **📌 Nota metodológica (empate técnico).** *RandomForest se eligió como modelo de
> producción por liderar en F1(clase=1), métrica principal del proyecto. Sin embargo,
> XGBoost obtiene mayor PR-AUC y Recall, métricas también relevantes en clasificación
> con clases desbalanceadas. La diferencia en F1 entre ambos (Δ=0.0102) es
> estadísticamente muy pequeña; en producción se recomendaría validar la elección con
> un segundo split o con bootstrap. Este proyecto opta por RF por su mejor
> interpretabilidad pedagógica.*

Reporte académico completo: **`reports/report.md`** y **`reports/report.pdf`**.

### Generar el PDF del reporte

```bash
python reports/build_pdf.py    # report.md → report.html → report.pdf (vía Chromium)
```

Si el conversor por Chromium fallara en tu entorno, alternativas equivalentes:

```bash
# Opción A — pandoc (si está instalado)
pandoc reports/report.md -o reports/report.pdf
# Opción B — md-to-pdf (Node)
npx md-to-pdf reports/report.md
```

El `.md` es la fuente canónica y es legible tal cual; el `.pdf` es un derivado.

## Reproducibilidad

- Semilla global: `RANDOM_STATE = 42` (split, CV y los 3 modelos).
- *Split* **estratificado 80/20** por `churn`; preprocesamiento ajustado **solo con train** (sin *data leakage*).
- Pipeline serializado completo (preprocesamiento + modelo) en `models/best_model.pkl`.
- Sin dependencias de runtime fuera de `requirements.txt`.
- Los módulos en `src/` son funciones puras: mismas entradas → mismas salidas.

## Hallazgos preliminares (resumen ejecutivo del notebook 01)

- Dataset: **7,043 clientes**, **21 variables**, ~**26.5% de churn** (desbalanceado).
- Solo se detectaron **11 valores faltantes implícitos** (en `TotalCharges`,
  todos correspondientes a clientes con `tenure = 0`): imputados con 0.0.
- **Ninguna variable numérica es normal** (Shapiro-Wilk p ≈ 0) → usamos métodos
  no paramétricos (Spearman, Mann-Whitney) además de Welch's t-test.
- Variables más asociadas a churn (chi² + Cramér's V):
  1. `contract` (V = 0.41) — confirma H1
  2. `tenure_group` (V = 0.35)
  3. `online_security` (V = 0.35)
  4. `tech_support` (V = 0.34)
  5. `internet_service` (V = 0.32)
- `gender` y `phone_service` **no** muestran asociación con churn.
- Multicolinealidad detectada entre `tenure` y `total_charges` (r ≈ 0.83), a
  manejar en feature selection.

## Diccionario de variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| customerID | str | Identificador único |
| gender | cat | Female / Male |
| SeniorCitizen | bin | ¿Mayor de 65? (0/1, mapeado a No/Yes) |
| Partner / Dependents | cat | Pareja / dependientes |
| tenure | num | Meses como cliente |
| PhoneService | cat | Servicio telefónico |
| MultipleLines | cat | Múltiples líneas |
| InternetService | cat | DSL / Fiber optic / No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport | cat | Servicios adicionales |
| StreamingTV, StreamingMovies | cat | Streaming |
| Contract | cat | Month-to-month / One year / Two year |
| PaperlessBilling | cat | Facturación digital |
| PaymentMethod | cat | Método de pago |
| MonthlyCharges | num | Cargo mensual ($) |
| TotalCharges | num | Cargo acumulado ($) |
| **Churn** | **bin** | **Target: ¿se fue? (Yes/No → 1/0)** |

## Verificación — reproducir el proyecto desde cero

Checklist de **5 pasos** que un estudiante debe correr (desde la raíz
`customer_churn_project/`) para reproducir todo el proyecto:

```bash
# 1) Instalar todas las dependencias
pip install -r requirements.txt

# 2) Ejecutar el notebook 01: ETL + EDA + pruebas estadísticas
#    (genera data/processed/churn_clean.csv y las figuras de EDA)
jupyter nbconvert --to notebook --execute notebooks/01_etl_eda.ipynb

# 3) Ejecutar el notebook 02: modelado, CV, evaluación, SHAP
#    (genera models/best_model.pkl, metadata.json, model_comparison.json y figuras)
jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb

# 4) Lanzar el dashboard (http://localhost:8501)
streamlit run app/dashboard.py

# 5) (Opcional) Validación visual del dashboard con capturas reales de navegador
python -m playwright install chromium
#    luego ejecutar el script de captura usado en la validación (reports/figures/dashboard/)
```

**Resultado esperado:** dataset limpio de 7,043 clientes, modelo ganador
**RandomForest** con **F1 ≈ 0.631 / recall ≈ 0.778 / ROC-AUC ≈ 0.843** en test,
dashboard funcional de 6 páginas y reporte en `reports/report.md` / `report.pdf`.

> Reproducibilidad garantizada por `RANDOM_STATE = 42`. Nota: el empate técnico
> RF↔XGBoost (Δ F1 = 0.0102) está documentado en `reports/report.md` §5.5.

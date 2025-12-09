# Credit Card Fraud Detection — Luis Goñi  

Análisis Exploratorio, Modelos Baseline y Evaluación Experimental 

Este repositorio contiene un estudio reproducible de detección de fraude en tarjetas de crédito utilizando el dataset público **Credit Card Fraud Detection Dataset (ULB Machine Learning Group)**, uno de los benchmarks más utilizados en investigación para problemas de clasificación altamente desbalanceados.

---

##  Objetivo del Proyecto

Desarrollar, evaluar y comparar distintos modelos de Machine Learning para detectar transacciones fraudulentas, poniendo especial foco en:

- Manejo del **desbalance extremo** (solo ~0.17% de fraudes) 
- Métricas apropiadas para fraude: **Recall**, **Precision**, **F1**, **ROC-AUC**, **PR-AUC** 
- Comparación de modelos base vs. modelos más robustos 
- Reproducibilidad del proceso mediante notebooks ordenados y código claro 

Este trabajo sirve como base para un **paper técnico**, una presentación profesional o una implementación industrial de un sistema antifraude.

---

## Dataset


El dataset utilizado es el clásico **Credit Card Fraud Detection Dataset** (Kaggle, ULB Machine Learning Group):

- 284.807 transacciones
- 492 fraudes (~0.17%)
- Variables V1 – V28 generadas mediante PCA
- Variables adicionales: Time y Amount

El dataset *no puede ser incluido directamente por restricciones de Kaggle*, por lo que debe descargarse desde:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


## Notebooks

notebooks/
│
├── 01_EDA.ipynb
│ - Exploratory Data Analysis
│ - Distribución del fraude
│ - Gráficos de Amount y Time
│ - Matriz de correlación
│
├── 02_baseline_models.ipynb
│ - Logistic Regression (baseline)
│ - Random Forest
│ - LinearSVC Calibrated (SVM escalable)
│ - Comparación PR-AUC
│ - Tabla comparativa de métricas
│
├── 03_resampling_models.ipynb
│ - Undersampling, Oversampling, SMOTE
│ - Mejora en recall de fraude
│
└── 04_threshold_tuning.ipynb
- Cambio del umbral de clasificación
- Optimización entre precision y recall

## Estructura del Proyecto

creditcard-fraud-detection-luis-goni/
│
├── data/
│ └── creditcard.csv # Necesita ser descargado desde Kaggle
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_baseline_models.ipynb
│ ├── 03_resampling_models.ipynb
│ └── 04_threshold_tuning.ipynb
│
├── src/
│ ├── data_preparation.py
│ ├── modeling.py
│ └── evaluation.py
│
├── figures/
│ ├── pr_curves.png
│ ├── roc_curves.png
│ └── metrics_table.png
│
└── README.md



## Resultados Principales

**Comparación de modelos (Baseline):**

| Modelo                     | ROC-AUC | PR-AUC | Precision Fraude | Recall Fraude | F1 Fraude |
|---------------------------|---------|--------|------------------|----------------|------------|
| Logistic Regression       | 0.956   | 0.707  | 0.858            | 0.615          | 0.717      |
| Random Forest             | 0.936   | 0.827  | 0.967            | 0.784          | 0.866      |
| LinearSVC Calibrated      | 0.968   | 0.703  | 0.854            | 0.554          | 0.672      |

**Random Forest es el mejor modelo** para este caso, logrando el balance óptimo entre recall y precisión de fraude.

## Visualizaciones

Las figuras principales se encuentran en `/figures/`:

- Precision–Recall Curve comparativa 
- ROC-AUC Curve 
- Tabla de métricas 

---

## Conclusiones

- El dataset presenta un fuerte desbalance (0.17% de fraude), por lo que métricas como **accuracy no son apropiadas**.
- Los modelos deben evaluarse con **PR-AUC**, **F1 fraude**, y **recall**.
- **Random Forest** es el modelo más robusto en este caso.
- LinearSVC calibrado es una alternativa escalable, pero con menor recall en fraude.
- Logistic Regression sirve como baseline interpretativo.
- Resampling y threshold tuning pueden mejorar el rendimiento en producción.

---

## Contacto

**Luis Goñi** 
Master’s student in Data Science 
LinkedIn: *(https://www.linkedin.com/in/luisgoni)* 



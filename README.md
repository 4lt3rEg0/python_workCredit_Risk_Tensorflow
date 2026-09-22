# Credit Risk Classification with TensorFlow

Proyecto de **Machine Learning / Deep Learning** para clasificación binaria de riesgo crediticio utilizando el **German Credit Data** del UCI Machine Learning Repository.

El objetivo es predecir si una solicitud representa **buen o mal riesgo crediticio** a partir de variables demográficas y financieras.

## Stack

- Python
- TensorFlow / Keras
- pandas / NumPy
- scikit-learn
- imbalanced-learn
- SHAP / LIME
- Matplotlib / Plotly
- Jupyter

## Pipeline

1. Descarga y carga del German Credit Data.
2. Separación train / validation / test.
3. Codificación one-hot de variables categóricas.
4. Escalado MinMax de variables numéricas.
5. Tratamiento del desbalance mediante SMOTE.
6. Entrenamiento de una red neuronal binaria.
7. Evaluación mediante accuracy, precision, recall y ROC-AUC.
8. Persistencia de modelos, preprocesadores e historial de entrenamiento.

## Arquitectura configurada

```text
Input
  ↓
Dense 64
  ↓
Dense 32
  ↓
Dense 16
  ↓
Sigmoid
```

La configuración incluye dropout, batch normalization, early stopping, reducción adaptativa del learning rate y checkpoint del mejor modelo.

## Ejemplo de resultado registrado

Uno de los entrenamientos almacenados en el repositorio registró sobre validación:

| Métrica | Valor |
| --- | ---: |
| Accuracy | 0.829 |
| Precision | 0.785 |
| Recall | 0.905 |
| ROC-AUC | 0.906 |

Estos valores corresponden a un artefacto de entrenamiento concreto y no deben interpretarse como una estimación universal del rendimiento del modelo.

## Estructura

```text
config/
  config.yaml
  params.yaml
data/
  models/
  processed/
notebooks/
  01_eda_analysis.ipynb
reports/
  figures/
src/
  data_preprocessing.py
  model_architecture.py
  train.py
  evaluate.py
main.py
evaluate_results.py
requirements.txt
```

## Instalación

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Entrenamiento

```bash
python main.py
```

## Evaluación

```bash
python evaluate_results.py
```

## Dataset

**German Credit Data — UCI Machine Learning Repository**

- 1.000 registros
- 20 variables predictoras
- clasificación binaria de riesgo
- distribución original aproximada: 70 % buen riesgo / 30 % mal riesgo

## Objetivo del proyecto

Proyecto académico orientado a practicar un pipeline completo de Deep Learning: preparación de datos, desbalance de clases, diseño y entrenamiento de redes neuronales, evaluación, persistencia de artefactos e interpretabilidad.

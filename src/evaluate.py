import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, accuracy_score)
import shap
import json
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def evaluate_model(self, model, X_test, y_test, save_results=True):
        """Evalúa el modelo en el conjunto de prueba"""
        logger.info("Evaluando modelo...")

        # 1. Predicciones
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # 2. Métricas principales
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # 3. Mostrar resultados
        print("\n" + "=" * 60)
        print("RESULTADOS DE EVALUACIÓN")
        print("=" * 60)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"Average Precision: {results['avg_precision']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Buen Riesgo', 'Mal Riesgo']))

        # 4. Generar visualizaciones
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba, results['roc_auc'])
        self.plot_precision_recall_curve(y_test, y_pred_proba, results['avg_precision'])

        # 5. Guardar resultados si se solicita
        if save_results:
            self.save_evaluation_results(results, y_test, y_pred, y_pred_proba)

        return results

    def plot_confusion_matrix(self, y_true, y_pred):
        """Genera y muestra la matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Bueno', 'Pred Malo'],
                    yticklabels=['Real Bueno', 'Real Malo'])
        plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
        plt.ylabel('Etiqueta Real', fontsize=12)
        plt.xlabel('Etiqueta Predicha', fontsize=12)

        # Guardar figura
        save_path = "./reports/figures/confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Matriz de confusión guardada en {save_path}")

    def plot_roc_curve(self, y_true, y_pred_proba, roc_auc):
        """Genera y muestra la curva ROC"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('Curva ROC', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Guardar figura
        save_path = "./reports/figures/roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_pred_proba, avg_precision):
        """Genera y muestra la curva Precision-Recall"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                 label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Curva Precision-Recall', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        # Guardar figura
        save_path = "./reports/figures/precision_recall_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_evaluation_results(self, results, y_true, y_pred, y_pred_proba):
        """Guarda los resultados de evaluación"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./reports/evaluation_{timestamp}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Guardar métricas como JSON
        with open(f"{save_dir}/metrics.json", "w") as f:
            json.dump(results, f, indent=4)

        # Guardar predicciones como CSV
        predictions_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'predicted_probability': y_pred_proba
        })
        predictions_df.to_csv(f"{save_dir}/predictions.csv", index=False)

        logger.info(f"Resultados guardados en {save_dir}")

    def explain_model(self, model, X_train, X_test, feature_names=None):
        """Explica el modelo usando SHAP (opcional)"""
        logger.info("Generando explicaciones SHAP...")

        # Crear explainer
        explainer = shap.Explainer(model, X_train[:100])  # Usar subset para velocidad

        # Calcular SHAP values
        shap_values = explainer(X_test[:50])  # Usar subset

        # Visualizar
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, show=False)
        plt.title('Importancia de Features (SHAP)', fontsize=14, fontweight='bold')

        # Guardar figura
        save_path = "./reports/figures/shap_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Explicaciones SHAP guardadas en {save_path}")
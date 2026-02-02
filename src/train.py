import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import yaml
import logging
from pathlib import Path

from data_preprocessing import DataPreprocessor
from model_architecture import CreditRiskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path="config/config.yaml", params_path="config/params.yaml"):
        self.config_path = config_path
        self.params_path = params_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Crear directorios necesarios
        self.create_directories()

    def create_directories(self):
        """Crea los directorios necesarios"""
        dirs = [
            self.config['data']['processed_path'],
            self.config['data']['model_path'],
            './logs',
            './reports',
            './reports/figures'
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def train(self):
        """Pipeline completo de entrenamiento"""
        logger.info("Iniciando pipeline de entrenamiento...")

        # 1. Preprocesamiento de datos
        preprocessor = DataPreprocessor(self.config_path)
        X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.run_pipeline()

        # 2. Construir modelo
        input_dim = X_train.shape[1]
        model_builder = CreditRiskModel(input_dim, self.params_path)
        model = model_builder.build_model()
        model = model_builder.compile_model()
        callbacks = model_builder.get_callbacks()

        # 3. Entrenar modelo
        logger.info("Iniciando entrenamiento...")
        training_params = yaml.safe_load(open(self.params_path))['model']['training']

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_params['epochs'],
            batch_size=training_params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        # 4. Guardar historia del entrenamiento
        self.save_training_history(history)

        # 5. Visualizar resultados
        self.plot_training_history(history)

        # 6. Guardar modelo final
        self.save_model(model, history)

        logger.info("Entrenamiento completado!")

        return model, history, (X_test, y_test)

    def save_training_history(self, history):
        """Guarda el historial de entrenamiento"""
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{self.config['data']['model_path']}/training_history_{timestamp}.json"

        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=4)

        logger.info(f"Historial guardado en {save_path}")

    def plot_training_history(self, history):
        """Genera gráficas del entrenamiento"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            ax.plot(history.history[metric], label=f'Train {metric}')
            ax.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            ax.set_title(f'{metric.capitalize()} por Época')
            ax.set_xlabel('Época')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Último subplot para learning rate si está disponible
        if 'lr' in history.history:
            ax = axes[-1]
            ax.plot(history.history['lr'])
            ax.set_title('Learning Rate por Época')
            ax.set_xlabel('Época')
            ax.set_ylabel('Learning Rate')
            ax.grid(True, alpha=0.3)
        else:
            axes[-1].axis('off')

        plt.tight_layout()

        # Guardar figura
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./reports/figures/training_history_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Gráficas guardadas en {save_path}")

    def save_model(self, model, history):
        """Guarda el modelo y metadatos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"{self.config['data']['model_path']}/model_{timestamp}"
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Guardar modelo completo
        model.save(f"{model_dir}/model.h5")

        # Guardar arquitectura como JSON
        model_json = model.to_json()
        with open(f"{model_dir}/architecture.json", "w") as json_file:
            json_file.write(model_json)

        # Guardar pesos
        model.save_weights(f"{model_dir}/weights.h5")

        # Guardar metadatos
        metadata = {
            "timestamp": timestamp,
            "input_shape": model.input_shape[1:],
            "output_shape": model.output_shape[1:],
            "total_params": model.count_params(),
            "training_history": {
                key: [float(v) for v in values[-1:]]  # Último valor de cada métrica
                for key, values in history.history.items()
            }
        }

        with open(f"{model_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Modelo guardado en {model_dir}")
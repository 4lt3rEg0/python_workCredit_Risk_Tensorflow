#!/usr/bin/env python3
"""
Punto de entrada principal del proyecto de Riesgo Crediticio
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='Sistema de Predicción de Riesgo Crediticio')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='Modo de ejecución')
    parser.add_argument('--model-path', type=str,
                        help='Ruta al modelo para evaluación/predicción')
    parser.add_argument('--data-path', type=str,
                        help='Ruta a los datos para predicción')

    args = parser.parse_args()

    # Configurar TensorFlow para usar GPU si está disponible
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU disponible: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No se detectaron GPUs, usando CPU")

    if args.mode == 'train':
        print("\n" + "=" * 60)
        print("MODO: ENTRENAMIENTO")
        print("=" * 60)

        trainer = ModelTrainer()
        model, history, test_data = trainer.train()

        # Evaluar inmediatamente después de entrenar
        print("\n" + "=" * 60)
        print("EVALUANDO MODELO RECIÉN ENTRENADO")
        print("=" * 60)

        evaluator = ModelEvaluator()
        X_test, y_test = test_data
        results = evaluator.evaluate_model(model, X_test, y_test)

        print("\n¡Proceso completado!")

    elif args.mode == 'evaluate':
        print("\n" + "=" * 60)
        print("MODO: EVALUACIÓN")
        print("=" * 60)

        if not args.model_path:
            print("ERROR: Debes especificar --model-path para evaluación")
            sys.exit(1)

        # Cargar modelo y datos
        model = tf.keras.models.load_model(args.model_path)

        # Aquí cargarías los datos de prueba
        # evaluator = ModelEvaluator()
        # results = evaluator.evaluate_model(model, X_test, y_test)

        print("Modo evaluación (implementación pendiente)")

    elif args.mode == 'predict':
        print("\n" + "=" * 60)
        print("MODO: PREDICCIÓN")
        print("=" * 60)

        if not args.model_path or not args.data_path:
            print("ERROR: Debes especificar --model-path y --data-path para predicción")
            sys.exit(1)

        print("Modo predicción (implementación pendiente)")


if __name__ == "__main__":
    main()
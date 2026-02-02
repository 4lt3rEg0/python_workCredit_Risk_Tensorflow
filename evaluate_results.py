# evaluate_results.py
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

print("=" * 60)
print("EVALUACIÓN DE RESULTADOS DEL ENTRENAMIENTO")
print("=" * 60)

# 1. Encontrar el último modelo entrenado
model_files = glob.glob("data/models/model_*/model.h5") + glob.glob("data/models/best_model*.h5")
if model_files:
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Último modelo: {latest_model}")

    # Cargar modelo
    model = tf.keras.models.load_model(latest_model)

    # 2. Cargar datos de prueba
    print("\nCargando datos para evaluación...")
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    _, _, X_test, _, _, y_test, _ = preprocessor.run_pipeline()

    # 3. Evaluar
    print("\nEvaluando modelo...")
    results = model.evaluate(X_test, y_test, verbose=0)

    print("\n📊 RESULTADOS:")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")

    # 4. Predicciones
    y_pred = model.predict(X_test, verbose=0)

    # 5. Ver historial de entrenamiento
    history_files = glob.glob("data/models/training_history*.json")
    if history_files:
        latest_history = max(history_files, key=os.path.getctime)
        print(f"\n📈 Historial de entrenamiento: {latest_history}")

        with open(latest_history, 'r') as f:
            history = json.load(f)

        # Graficar
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Pérdida
        axes[0].plot(history['loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(history['accuracy'], label='Train')
        axes[1].plot(history['val_accuracy'], label='Validation')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
else:
    print("No se encontraron modelos entrenados")
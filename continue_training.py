# continue_training.py - VERSIÓN CORREGIDA
import sys
import os
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
import tensorflow as tf
import yaml
import numpy as np

print("=" * 60)
print("CONTINUANDO ENTRENAMIENTO DESDE CHECKPOINT")
print("=" * 60)

# 1. Cargar configuración
with open("config/params.yaml", 'r') as f:
    params = yaml.safe_load(f)

# 2. Cargar datos
print("\n1. Cargando datos...")
preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.run_pipeline()

print(f"   ✓ Datos cargados: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# 3. Cargar el modelo guardado
print("\n2. Cargando modelo desde checkpoint...")
try:
    # Primero buscar el mejor modelo
    import glob

    model_files = glob.glob("data/models/best_model*.h5") + glob.glob("data/models/model_*/model.h5")

    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"   ✓ Cargando: {latest_model}")
        model = tf.keras.models.load_model(latest_model)
        print(f"   ✓ Modelo cargado: {model.input_shape} -> {model.output_shape}")
        print(f"   ✓ Parámetros: {model.count_params():,}")
    else:
        print("   ⚠ No se encontraron modelos guardados, creando nuevo...")
        from model_architecture import CreditRiskModel

        model_builder = CreditRiskModel(X_train.shape[1])
        model = model_builder.build_model()
        model = model_builder.compile_model()

except Exception as e:
    print(f"   ✗ Error cargando modelo: {e}")
    print("   Creando nuevo modelo...")
    from model_architecture import CreditRiskModel

    model_builder = CreditRiskModel(X_train.shape[1])
    model = model_builder.build_model()
    model = model_builder.compile_model()

# 4. Configurar callbacks para continuar
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=params['model']['callbacks']['early_stopping']['patience'],
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="data/models/best_model_continued.h5",
        monitor='val_auc',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        "data/models/training_log.csv",
        separator=",",
        append=True
    )
]

# 5. Determinar epoch inicial
# Buscar si hay log de entrenamiento previo
initial_epoch = 0
if os.path.exists("data/models/training_log.csv"):
    import pandas as pd

    try:
        log_df = pd.read_csv("data/models/training_log.csv")
        if 'epoch' in log_df.columns:
            initial_epoch = log_df['epoch'].max() + 1
            print(f"   ✓ Continuando desde epoch: {initial_epoch}")
    except:
        initial_epoch = 0

print(f"\n3. Continuando entrenamiento desde epoch {initial_epoch}...")
print(f"   Épocas adicionales: 50")
print(f"   Batch size: {params['model']['training']['batch_size']}")

# 6. Continuar entrenamiento
try:
    history_continued = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=initial_epoch + 50,  # Entrenar hasta epoch+50
        batch_size=params['model']['training']['batch_size'],
        callbacks=callbacks,
        verbose=1,
        initial_epoch=initial_epoch  # CORREGIDO: usar initial_epoch calculado
    )

    print("\n✅ Entrenamiento continuado completado")
    print(f"   Modelo guardado en: data/models/best_model_continued.h5")

    # 7. Evaluar
    print("\n4. Evaluando modelo actualizado...")
    results = model.evaluate(X_test, y_test, verbose=0)

    print("\n📊 RESULTADOS ACTUALIZADOS:")
    metrics = model.metrics_names
    for name, value in zip(metrics, results):
        print(f"   {name:15}: {value:.4f}")

    # 8. Guardar historial completo
    import json
    from datetime import datetime

    if hasattr(history_continued, 'history'):
        history_dict = history_continued.history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"data/models/training_history_continued_{timestamp}.json"

        with open(save_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history_dict.items()}, f, indent=4)

        print(f"\n📈 Historial guardado en: {save_path}")

except Exception as e:
    print(f"\n❌ Error durante el entrenamiento: {e}")
    import traceback

    traceback.print_exc()
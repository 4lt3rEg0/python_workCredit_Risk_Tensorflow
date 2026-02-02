import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    def __init__(self, input_dim, config_path="config/params.yaml"):
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        self.input_dim = input_dim
        self.model = None

    def build_model(self):
        """Construye la arquitectura del modelo"""
        logger.info(f"Construyendo modelo con input_dim={self.input_dim}")

        model_params = self.params['model']['architecture']

        # Capa de entrada
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs

        # Capas ocultas
        for i, units in enumerate(model_params['hidden_layers']):
            x = layers.Dense(
                units,
                activation=model_params['activation'],
                kernel_regularizer=regularizers.l2(0.01),
                name=f"dense_{i + 1}"
            )(x)

            if model_params['use_batch_norm']:
                x = layers.BatchNormalization(name=f"batch_norm_{i + 1}")(x)

            x = layers.Dropout(
                model_params['dropout_rate'],
                name=f"dropout_{i + 1}"
            )(x)

        # Capa de atención (simplificada)
        if model_params['use_attention']:
            attention = layers.Dense(x.shape[-1], activation='tanh', name='attention')(x)
            attention = layers.Dense(1, activation='softmax', name='attention_weights')(attention)
            x = layers.Multiply(name='attention_applied')([x, attention])

        # Capa de salida
        outputs = layers.Dense(
            1,
            activation=model_params['output_activation'],
            name='output'
        )(x)

        # Crear modelo
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='CreditRiskModel')

        self.model.summary()
        return self.model

    def compile_model(self):
        """Compila el modelo con los parámetros de configuración"""
        training_params = self.params['model']['training']

        optimizer_map = {
            'adam': keras.optimizers.Adam,
            'sgd': keras.optimizers.SGD,
            'rmsprop': keras.optimizers.RMSprop
        }

        optimizer_class = optimizer_map.get(training_params['optimizer'], keras.optimizers.Adam)
        optimizer = optimizer_class(learning_rate=training_params['learning_rate'])

        # Definir métricas
        metrics = []
        for metric_name in training_params['metrics']:
            if metric_name == 'accuracy':
                metrics.append('accuracy')
            elif metric_name == 'precision':
                metrics.append(keras.metrics.Precision(name='precision'))
            elif metric_name == 'recall':
                metrics.append(keras.metrics.Recall(name='recall'))
            elif metric_name == 'auc':
                metrics.append(keras.metrics.AUC(name='auc'))

        self.model.compile(
            optimizer=optimizer,
            loss=training_params['loss'],
            metrics=metrics
        )

        logger.info("Modelo compilado")
        return self.model

    def get_callbacks(self):
        """Crea los callbacks para el entrenamiento"""
        callback_params = self.params['model']['callbacks']
        callbacks_list = []

        # Early Stopping
        if callback_params['early_stopping']['monitor']:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=callback_params['early_stopping']['monitor'],
                patience=callback_params['early_stopping']['patience'],
                mode=callback_params['early_stopping']['mode'],
                restore_best_weights=callback_params['early_stopping']['restore_best_weights'],
                verbose=1
            )
            callbacks_list.append(early_stopping)

        # Reduce Learning Rate
        if callback_params['reduce_lr']['monitor']:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor=callback_params['reduce_lr']['monitor'],
                factor=callback_params['reduce_lr']['factor'],
                patience=callback_params['reduce_lr']['patience'],
                min_lr=callback_params['reduce_lr']['min_lr'],
                verbose=1
            )
            callbacks_list.append(reduce_lr)

        # Model Checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=callback_params['checkpoint']['filepath'],
            monitor=callback_params['checkpoint']['monitor'],
            save_best_only=callback_params['checkpoint']['save_best_only'],
            verbose=1
        )
        callbacks_list.append(checkpoint)

        # TensorBoard (opcional)
        tensorboard = keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )
        callbacks_list.append(tensorboard)

        return callbacks_list
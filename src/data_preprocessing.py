import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml
import logging
import joblib
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.preprocessor = None
        self.scaler = None
        self.feature_names = None

    def load_data(self):
        """Carga el dataset German Credit"""
        logger.info("Cargando German Credit Data...")

        df = pd.read_csv(
            self.config['data']['german_credit']['url'],
            delimiter=' ',
            names=self.config['data']['german_credit']['columns'],
            header=None
        )

        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df

    def prepare_target(self, df):
        """Prepara la variable objetivo"""
        target_col = 'class'
        # Recodificar según configuración
        encoding_map = self.config['preprocessing']['target_encoding']
        df[target_col] = df[target_col].map(encoding_map)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        logger.info(f"Distribución de clases: {y.value_counts().to_dict()}")
        logger.info(f"Proporciones: {y.value_counts(normalize=True).to_dict()}")
        return X, y

    def get_onehot_encoder(self):
        """Devuelve un OneHotEncoder compatible con cualquier versión de scikit-learn"""
        try:
            # Intentar con scikit-learn >= 1.2
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            logger.info("Usando OneHotEncoder con sparse_output=False (scikit-learn >= 1.2)")
            return encoder
        except TypeError:
            try:
                # Intentar con scikit-learn < 1.2
                encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                logger.info("Usando OneHotEncoder con sparse=False (scikit-learn < 1.2)")
                return encoder
            except TypeError:
                # Último recurso
                encoder = OneHotEncoder(handle_unknown='ignore')
                logger.info("Usando OneHotEncoder con parámetros por defecto")
                return encoder

    def create_preprocessor(self, X):
        """Crea el pipeline de preprocesamiento - VERSIÓN CORREGIDA"""
        # Identificar tipos de columnas
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        logger.info(f"Columnas categóricas ({len(categorical_cols)}): {categorical_cols}")
        logger.info(f"Columnas numéricas ({len(numerical_cols)}): {numerical_cols}")

        # Pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        # Usar el método compatible con versiones
        onehot_encoder = self.get_onehot_encoder()

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', onehot_encoder)
        ])

        # ColumnTransformer
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ], remainder='drop')

        return self.preprocessor

    def balance_data(self, X, y):
        """Aplica SMOTE para balancear clases"""
        logger.info("Aplicando SMOTE para balancear clases...")

        # Convertir y a numpy array si no lo es
        y_array = np.array(y) if not isinstance(y, np.ndarray) else y

        logger.info(f"Antes SMOTE: Clase 0={np.sum(y_array == 0)}, Clase 1={np.sum(y_array == 1)}")

        smote = SMOTE(random_state=self.config['preprocessing']['random_state'])
        X_bal, y_bal = smote.fit_resample(X, y_array)

        logger.info(f"Después SMOTE: Clase 0={np.sum(y_bal == 0)}, Clase 1={np.sum(y_bal == 1)}")

        return X_bal, y_bal

    def split_data(self, X, y):
        """Divide en train, validation, test"""
        test_size = self.config['preprocessing']['test_size']
        val_size = self.config['preprocessing']['validation_size']

        # Primera división: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.config['preprocessing']['random_state'],
            stratify=y
        )

        # Segunda división: train / val
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative,
            random_state=self.config['preprocessing']['random_state'],
            stratify=y_temp
        )

        logger.info(f"Train: {X_train.shape} ({X_train.shape[0] / len(X) * 100:.1f}%)")
        logger.info(f"Val: {X_val.shape} ({X_val.shape[0] / len(X) * 100:.1f}%)")
        logger.info(f"Test: {X_test.shape} ({X_test.shape[0] / len(X) * 100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def run_pipeline(self):
        """Ejecuta todo el pipeline de preprocesamiento"""
        logger.info("=" * 50)
        logger.info("INICIANDO PIPELINE DE PREPROCESAMIENTO")
        logger.info("=" * 50)

        # 1. Cargar datos
        df = self.load_data()

        # 2. Preparar variables
        X, y = self.prepare_target(df)

        # 3. Crear preprocesador
        preprocessor = self.create_preprocessor(X)

        # 4. Aplicar transformaciones
        logger.info("Aplicando transformaciones...")
        X_processed = preprocessor.fit_transform(X)

        # Guardar nombres de características
        try:
            self.feature_names = preprocessor.get_feature_names_out()
            logger.info(f"Número de características: {len(self.feature_names)}")
        except AttributeError:
            logger.warning("No se pudieron obtener nombres de características")
            self.feature_names = None

        logger.info(f"X procesado: {X_processed.shape} (tipo: {type(X_processed)})")

        # 5. Balancear datos
        X_bal, y_bal = self.balance_data(X_processed, y)

        # 6. Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_bal, y_bal)

        # 7. Guardar preprocesador para uso futuro
        self.save_preprocessor(preprocessor)

        # 8. Guardar nombres de características
        if self.feature_names is not None:
            self.save_feature_names()

        logger.info("✓ Pipeline completado exitosamente")
        return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

    def save_preprocessor(self, preprocessor):
        """Guarda el preprocesador para uso futuro"""
        save_path = self.config['data']['processed_path']
        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}/preprocessor_{timestamp}.pkl"

        joblib.dump(preprocessor, filename)
        logger.info(f"Preprocesador guardado en {filename}")

    def save_feature_names(self):
        """Guarda los nombres de las características"""
        if self.feature_names is not None:
            save_path = self.config['data']['processed_path']
            os.makedirs(save_path, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_path}/feature_names_{timestamp}.txt"

            with open(filename, 'w') as f:
                for name in self.feature_names:
                    f.write(f"{name}\n")

            logger.info(f"Nombres de características guardados en {filename}")
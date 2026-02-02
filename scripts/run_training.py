#!/usr/bin/env python3
"""
Script para ejecutar el entrenamiento desde la línea de comandos
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import ModelTrainer

if __name__ == "__main__":
    print("Ejecutando entrenamiento del modelo...")
    trainer = ModelTrainer()
    model, history, test_data = trainer.train()
    print("\nEntrenamiento completado exitosamente!")
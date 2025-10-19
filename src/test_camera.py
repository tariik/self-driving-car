#!/usr/bin/env python
"""
Script de prueba para verificar qué captura el sensor de cámara
Guarda una imagen de prueba para visualización
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.agents.drl_flatten_agent import DRLFlattenAgent
from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv


def main():
    print("🔍 Test de Sensor de Cámara")
    print("=" * 50)
    
    # Configuración
    config = {
        "carla": {
            "host": "localhost",
            "port": 3000,
            "timeout": 60.0,
            "timestep": 0.1,
            "retries_on_error": 10,
            "resolution_x": 800,
            "resolution_y": 600,
            "quality_level": "Low",
            "enable_rendering": True,
            "show_display": False,
            "use_external_server": True
        },
        "others": {
            "framestack": 4,
            "max_time_idle": 100,
            "max_time_episode": 500,
        }
    }
    
    # Configuración del experimento con cámara trasera
    experiment_config = BASE_EXPERIMENT_CONFIG.copy()
    experiment_config["hero"]["sensors"] = {
        "rgb_camera": {
            "type": "sensor.camera.rgb",
            "transform": "-5.5,0.0,2.8,0.0,-15.0,0.0",  # Detrás del vehículo
            "image_size_x": "84",
            "image_size_y": "84",
            "fov": "90",
            "size": 84
        }
    }
    experiment_config["hero"]["spawn_points"] = []
    experiment_config["town"] = "Town01"
    experiment_config["weather"] = "ClearNoon"
    experiment_config["others"] = config["others"]
    
    print("\n📷 Configuración de cámara:")
    print(f"   Posición: -5.5m detrás, 2.8m altura")
    print(f"   Ángulo: -15° (mirando hacia la carretera)")
    print(f"   Resolución: 84x84")
    print(f"   FOV: 90°")
    
    # Crear entorno
    print("\n🚗 Creando entorno...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    # Reset para obtener primera observación
    print("🔄 Reset del entorno...")
    state, _ = env.reset()
    state, _ = env.reset()  # Doble reset para estabilizar
    
    print(f"\n📊 Estado recibido:")
    print(f"   Shape: {state.shape}")
    print(f"   Type: {state.dtype}")
    print(f"   Min: {state.min()}, Max: {state.max()}")
    
    # Tomar una acción para obtener nueva imagen
    print("\n🎮 Ejecutando una acción (avanzar recto)...")
    action = 8  # Straight
    state, reward, done, _, _ = env.step(action)
    
    # Guardar las imágenes del frame stack
    output_dir = "../test_camera_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Guardando frames en: {output_dir}/")
    
    # Guardar cada frame del stack
    for i in range(state.shape[2]):
        frame = state[:, :, i]
        
        # Convertir a uint8 si es necesario
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Crear imagen en escala de grises
        img_gray = Image.fromarray(frame, mode='L')
        
        # Escalar para mejor visualización
        img_gray_large = img_gray.resize((336, 336), Image.NEAREST)
        img_gray_large.save(f"{output_dir}/frame_{i}_grayscale.png")
        
        # Convertir a RGB para visualización en color
        img_rgb = img_gray.convert('RGB')
        img_rgb_large = img_rgb.resize((336, 336), Image.NEAREST)
        img_rgb_large.save(f"{output_dir}/frame_{i}_rgb.png")
        
        print(f"   ✓ Frame {i}: {frame.shape}, min={frame.min()}, max={frame.max()}")
    
    # Crear imagen combinada con los 4 frames
    print("\n🖼️  Creando imagen combinada...")
    combined = Image.new('RGB', (336 * 2, 336 * 2))
    for i in range(4):
        frame = state[:, :, i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        img = Image.fromarray(frame, mode='L').convert('RGB')
        img = img.resize((336, 336), Image.NEAREST)
        x = (i % 2) * 336
        y = (i // 2) * 336
        combined.paste(img, (x, y))
    
    combined.save(f"{output_dir}/combined_4_frames.png")
    print(f"   ✓ Imagen combinada guardada")
    
    # Cerrar entorno
    env.close()
    
    print("\n" + "=" * 50)
    print("✅ Test completado!")
    print(f"\n📁 Ver imágenes:")
    print(f"   eog {output_dir}/frame_3_rgb.png")
    print(f"   eog {output_dir}/combined_4_frames.png")
    print()


if __name__ == "__main__":
    main()

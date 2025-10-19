#!/usr/bin/env python3
"""
Debug: Visualizar las imágenes en cada paso del procesamiento
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import numpy as np
from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv

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
        "framestack": 1,
        "max_time_idle": 100,
        "max_time_episode": 500,
    }
}

experiment_config = BASE_EXPERIMENT_CONFIG.copy()
experiment_config["use_random_routes"] = True
experiment_config["hero"]["sensors"] = {
    "rgb_camera": {
        "type": "sensor.camera.rgb",
        "transform": "1.5,0.0,1.5,0.0,0.0,0.0",
        "image_size_x": "640",
        "image_size_y": "480",
        "fov": "90",
        "size": 11
    }
}
experiment_config["hero"]["spawn_points"] = []
experiment_config["town"] = "Town01"
experiment_config["weather"] = "ClearNoon"
experiment_config["others"] = config["others"]

print("=" * 70)
print("🔍 DEBUG: Procesamiento de Imagen")
print("=" * 70)

try:
    print("\n🔧 Creando entorno...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    print("🔄 Reset...")
    state, _ = env.reset()
    
    # Capturar una imagen raw
    print("\n📸 Capturando imagen...")
    sensor_data = env.core.tick(None)
    
    if 'rgb_camera' in sensor_data:
        raw_image = sensor_data['rgb_camera'][1]
        
        print(f"\n✅ Imagen RAW de CARLA:")
        print(f"   - Shape: {raw_image.shape}")
        print(f"   - Dtype: {raw_image.dtype}")
        print(f"   - Min: {raw_image.min()}, Max: {raw_image.max()}")
        print(f"   - Mean: {raw_image.mean():.2f}")
        
        # Guardar imagen original
        cv2.imwrite('/tmp/debug_1_original_640x480.png', cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
        print(f"   💾 Guardada: /tmp/debug_1_original_640x480.png")
        
        # Convertir a grayscale
        gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
        print(f"\n✅ Después de RGB→Grayscale:")
        print(f"   - Shape: {gray.shape}")
        print(f"   - Dtype: {gray.dtype}")
        print(f"   - Min: {gray.min()}, Max: {gray.max()}")
        print(f"   - Mean: {gray.mean():.2f}")
        cv2.imwrite('/tmp/debug_2_grayscale_640x480.png', gray)
        print(f"   💾 Guardada: /tmp/debug_2_grayscale_640x480.png")
        
        # Resize a 11x11
        resized = cv2.resize(gray, (11, 11))
        print(f"\n✅ Después de Resize 11×11:")
        print(f"   - Shape: {resized.shape}")
        print(f"   - Dtype: {resized.dtype}")
        print(f"   - Min: {resized.min()}, Max: {resized.max()}")
        print(f"   - Mean: {resized.mean():.2f}")
        
        # Escalar para visualización
        resized_large = cv2.resize(resized, (220, 220), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('/tmp/debug_3_resized_11x11.png', resized_large)
        print(f"   💾 Guardada (escalada): /tmp/debug_3_resized_11x11.png")
        
        # Mostrar la matriz 11x11
        print(f"\n📊 Matriz 11×11 (valores de píxeles):")
        print(resized)
        
        # Normalizar [-1, 1]
        normalized = (resized.astype(np.float32) - 128) / 128
        print(f"\n✅ Después de Normalización [-1, 1]:")
        print(f"   - Shape: {normalized.shape}")
        print(f"   - Dtype: {normalized.dtype}")
        print(f"   - Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")
        print(f"   - Mean: {normalized.mean():.3f}")
        
        # Agregar dimensión de canal
        with_channel = normalized[:, :, np.newaxis]
        print(f"\n✅ Con dimensión de canal:")
        print(f"   - Shape: {with_channel.shape}")
        
        # Flatten
        flattened = with_channel.flatten()
        print(f"\n✅ Después de Flatten:")
        print(f"   - Shape: {flattened.shape}")
        print(f"   - Primeros 10 valores: {flattened[:10]}")
        
        print(f"\n✅ Estado final:")
        print(f"   - Shape: {state.shape}")
        print(f"   - Primeros 10 valores (imagen): {state[:10]}")
        print(f"   - Valores 122-123 (φt, dt): {state[121:]}")
        
    else:
        print("❌ No se encontró 'rgb_camera' en sensor_data")
        print(f"   Sensores disponibles: {list(sensor_data.keys())}")
    
    print("\n🧹 Limpiando...")
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ DEBUG COMPLETADO")
    print("=" * 70)
    print("\n📁 Imágenes guardadas en /tmp/:")
    print("   1. debug_1_original_640x480.png")
    print("   2. debug_2_grayscale_640x480.png")
    print("   3. debug_3_resized_11x11.png (escalada 220×220 para ver)")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

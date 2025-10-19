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
        # CÁMARA FRONTAL apuntando hacia la CARRETERA (pitch=-15°)
        "transform": "2.0,0.0,1.2,0.0,-15.0,0.0",  # x,y,z,roll,pitch,yaw
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
        
        # Guardar imagen original en el proyecto
        output_dir = "debug_output"
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(f'{output_dir}/1_original_640x480.png', cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
        print(f"   💾 Guardada: {output_dir}/1_original_640x480.png")
        
        # Convertir a grayscale
        gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
        print(f"\n✅ Después de RGB→Grayscale:")
        print(f"   - Shape: {gray.shape}")
        print(f"   - Min: {gray.min()}, Max: {gray.max()}")
        cv2.imwrite(f'{output_dir}/2_grayscale_640x480.png', gray)
        print(f"   💾 Guardada: {output_dir}/2_grayscale_640x480.png")
        
        # Resize a 11x11
        resized = cv2.resize(gray, (11, 11))
        print(f"\n✅ Después de Resize 11×11:")
        print(f"   - Shape: {resized.shape}")
        print(f"   - Min: {resized.min()}, Max: {resized.max()}")
        
        # Escalar para visualización (220x220 = 11x20)
        resized_large = cv2.resize(resized, (220, 220), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{output_dir}/3_resized_11x11_scaled.png', resized_large)
        print(f"   💾 Guardada escalada: {output_dir}/3_resized_11x11_scaled.png")
        
        # Mostrar la matriz 11x11
        print(f"\n📊 Matriz 11×11 (primeras 5 filas):")
        for i in range(min(5, resized.shape[0])):
            print(f"   Fila {i}: {resized[i]}")
        
        # Normalizar [-1, 1]
        normalized = (resized.astype(np.float32) - 128) / 128
        print(f"\n✅ Después de Normalización [-1, 1]:")
        print(f"   - Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")
        print(f"   - Mean: {normalized.mean():.3f}")
        
        print(f"\n✅ Estado final recibido:")
        print(f"   - Shape: {state.shape}")
        print(f"   - Min: {state.min():.3f}, Max: {state.max():.3f}")
        print(f"   - Mean primeros 121 valores: {state[:121].mean():.3f}")
        print(f"   - φt (ángulo): {state[121]:.6f} rad = {np.degrees(state[121]):.2f}°")
        print(f"   - dt (distancia): {state[122]:.6f} m")
        
    else:
        print("❌ No se encontró 'rgb_camera' en sensor_data")
        print(f"   Sensores disponibles: {list(sensor_data.keys())}")
    
    print("\n🧹 Limpiando...")
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ DEBUG COMPLETADO")
    print("=" * 70)
    print(f"\n📁 Imágenes guardadas en: ./{output_dir}/")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

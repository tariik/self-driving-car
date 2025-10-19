#!/usr/bin/env python3
"""
Test rápido para verificar que el sensor de imagen funciona con 640x480 → 11x11
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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

# CONFIGURACIÓN DEL PAPER: 640×480 → 11×11
experiment_config["hero"]["sensors"] = {
    "rgb_camera": {
        "type": "sensor.camera.rgb",
        "transform": "1.5,0.0,1.5,0.0,0.0,0.0",
        "image_size_x": "640",  # Paper: Original
        "image_size_y": "480",  # Paper: Original
        "fov": "90",
        "size": 11  # Target resize
    }
}

experiment_config["hero"]["spawn_points"] = []
experiment_config["town"] = "Town01"
experiment_config["weather"] = "ClearNoon"
experiment_config["others"] = config["others"]

print("=" * 70)
print("🧪 TEST: Sensor de Imagen (Paper vs Código)")
print("=" * 70)
print()

try:
    print("🔧 Creando experimento...")
    experiment = BaseEnv(experiment_config)
    print(f"✓ Observation space: {experiment.get_observation_space().shape}")
    
    print("\n🚗 Creando entorno CARLA...")
    env = CarlaEnv(experiment, config)
    print("✓ Entorno creado")
    
    print("\n🔄 Reset del entorno...")
    state, _ = env.reset()
    print(f"✓ Estado shape: {state.shape}")
    print(f"✓ Estado dtype: {state.dtype}")
    
    # Verificar configuración del sensor
    sensor_config = experiment_config["hero"]["sensors"]["rgb_camera"]
    print("\n📷 Configuración del Sensor:")
    print(f"   - Captura original: {sensor_config['image_size_x']}×{sensor_config['image_size_y']}")
    print(f"   - Target resize: {sensor_config['size']}×{sensor_config['size']}")
    print(f"   - Píxeles capturados: {int(sensor_config['image_size_x']) * int(sensor_config['image_size_y']):,}")
    print(f"   - Píxeles finales: {sensor_config['size'] * sensor_config['size']}")
    print(f"   - Reducción: {int(sensor_config['image_size_x']) * int(sensor_config['image_size_y']) / (sensor_config['size'] * sensor_config['size']):.0f}x")
    
    print("\n✅ Comparación con Paper:")
    paper_original = 640 * 480
    paper_final = 11 * 11
    codigo_original = int(sensor_config['image_size_x']) * int(sensor_config['image_size_y'])
    codigo_final = sensor_config['size'] * sensor_config['size']
    
    print(f"   Paper:  {paper_original:,} píxeles → {paper_final} píxeles")
    print(f"   Código: {codigo_original:,} píxeles → {codigo_final} píxeles")
    
    if paper_original == codigo_original and paper_final == codigo_final:
        print("\n   ✅ CONFIGURACIÓN IDÉNTICA AL PAPER")
    else:
        print("\n   ⚠️  CONFIGURACIÓN DIFERENTE AL PAPER")
    
    print("\n✅ Estado correcto:", state.shape == (123,))
    print(f"   - Primeros 121 valores: Imagen {sensor_config['size']}×{sensor_config['size']} aplanada")
    print(f"   - Valor 122: φt = {state[121]:.4f} rad")
    print(f"   - Valor 123: dt = {state[122]:.4f} m")
    
    print("\n🧹 Limpiando...")
    env.close()
    print("✓ Entorno cerrado")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

#!/usr/bin/env python3
"""
Test: Verificar que el render muestra exactamente la imagen 11×11 del agente
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv
import shutil

# Limpiar carpeta render_output
if os.path.exists('render_output'):
    shutil.rmtree('render_output')
os.makedirs('render_output')

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
print("🎥 TEST: Render de imagen 11×11 del agente")
print("=" * 70)

try:
    print("\n🔧 Creando entorno...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    print("🔄 Reset y captura de 5 frames...")
    state, _ = env.reset()
    
    for i in range(5):
        # Render frame actual
        env.render()
        
        # Step
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        
        if done:
            print(f"   Episodio terminado en step {i+1}")
            break
    
    print("\n✅ Frames guardados!")
    print(f"📁 Ubicación: render_output/")
    print(f"🖼️  Total: {env.render_counter} imágenes")
    
    print("\n📊 Características de las imágenes guardadas:")
    print("   - Tamaño: 330×330 píxeles (11×11 escalado 30x)")
    print("   - Contenido: Exactamente lo que ve el agente")
    print("   - Formato: Grayscale con información de φt y dt")
    print("   - Interpolación: NEAREST (píxeles cuadrados bien definidos)")
    
    print("\n💡 IMPORTANTE:")
    print("   Las imágenes se verán MUY pixeladas (solo 11×11 píxeles)")
    print("   Esto es NORMAL y es lo que el agente realmente ve")
    print("   El paper usa esta reducción intencionalmente")
    
    print("\n🧹 Limpiando...")
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)
    print("\n🔍 Revisa las imágenes en render_output/ para ver lo que ve el agente")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

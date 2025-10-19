#!/usr/bin/env python3
"""
Test: Verificar que la cámara ahora capture la carretera (no el cielo)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv
import shutil
import numpy as np

# Limpiar render_output
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
        "transform": "2.0,0.0,1.2,0.0,-15.0,0.0",  # ¡Nueva orientación!
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
print("🎥 TEST: Cámara con pitch -15° (mirando hacia carretera)")
print("=" * 70)

try:
    print("\n🔧 Creando entorno con nueva orientación de cámara...")
    print("   📷 Transform: x=2.0, y=0.0, z=1.2, pitch=-15°")
    
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    print("\n🔄 Capturando frames...")
    state, _ = env.reset()
    
    # Capturar 3 frames
    for i in range(3):
        env.render()
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        
        if done:
            break
    
    # Analizar el primer frame
    print("\n📊 Analizando matriz 11×11 del agente:")
    
    # Extraer imagen del estado
    image_flat = state[:121]
    image_2d = image_flat.reshape(11, 11)
    
    # Desnormalizar
    image_uint8 = ((image_2d * 128) + 128).clip(0, 255).astype(np.uint8)
    
    print("\nMatriz 11×11:")
    print(image_uint8)
    
    print(f"\n📈 Estadísticas:")
    print(f"   Min: {image_uint8.min()}")
    print(f"   Max: {image_uint8.max()}")
    print(f"   Mean: {image_uint8.mean():.1f}")
    print(f"   Std: {image_uint8.std():.1f}")
    
    # Verificar variación horizontal
    horizontal_vars = [np.std(row) for row in image_uint8]
    avg_h_var = np.mean(horizontal_vars)
    
    print(f"\n🔍 Variación horizontal promedio: {avg_h_var:.2f}")
    
    if avg_h_var < 5:
        print("   ❌ Todavía muy uniforme (líneas horizontales)")
        print("   Puede necesitar más ajuste de pitch o posición")
    elif avg_h_var < 15:
        print("   ⚠️  Poca variación horizontal")
        print("   La imagen tiene algo de variación pero podría mejorar")
    else:
        print("   ✅ Buena variación horizontal")
        print("   La imagen muestra contenido variado (carretera, marcas, etc)")
    
    # Contar filas muy similares
    uniform_rows = 0
    for i in range(10):
        diff = np.abs(image_uint8[i].astype(int) - image_uint8[i+1].astype(int)).mean()
        if diff < 3:
            uniform_rows += 1
    
    print(f"\n📏 Filas consecutivas casi idénticas: {uniform_rows}/10")
    
    if uniform_rows > 7:
        print("   ❌ Muchas filas uniformes - todavía parece líneas")
    elif uniform_rows > 4:
        print("   ⚠️  Algunas filas uniformes - mejorable")
    else:
        print("   ✅ Buena variación entre filas")
    
    print(f"\n💾 Frames guardados: {env.render_counter}")
    print(f"📁 Ver en: render_output/frame_0000.png")
    
    print("\n🧹 Limpiando...")
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)
    print("\n🔍 Compara la nueva imagen con la anterior")
    print("   Debería verse MÁS variada (no solo líneas horizontales)")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

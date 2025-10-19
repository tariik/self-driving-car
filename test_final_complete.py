#!/usr/bin/env python3
"""
TEST FINAL: Verificar que la configuración completa funciona correctamente
- Cámara con pitch=-15° (apuntando a carretera)
- Render activado (guardando imágenes)
- Imagen 11×11 con variación (no solo líneas horizontales)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv
from src.agents.drl_flatten_agent import DRLFlattenAgent
import shutil
import numpy as np
import time

SEED = 42

# Limpiar render_output
if os.path.exists('render_output'):
    shutil.rmtree('render_output')
os.makedirs('render_output')

print("=" * 70)
print("🎯 TEST FINAL: Verificación Completa del Sistema")
print("=" * 70)

# Usar la MISMA configuración que src/main.py
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

# ✅ Configuración EXACTA de src/main.py
experiment_config["hero"]["sensors"] = {
    "rgb_camera": {
        "type": "sensor.camera.rgb",
        "transform": "2.0,0.0,1.2,0.0,-15.0,0.0",  # pitch=-15°
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

print("\n📋 Configuración:")
print(f"   - Mapa: Town01")
print(f"   - Cámara: pitch=-15° (apuntando a carretera)")
print(f"   - Imagen: 640×480 → 11×11")
print(f"   - Estado: 123 dims (121 píxeles + φt + dt)")

try:
    print("\n🔧 Paso 1/5: Creando entorno...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    print("   ✅ Entorno creado")
    
    print("\n🔧 Paso 2/5: Creando agente DRL...")
    agent = DRLFlattenAgent(env=env, seed=SEED)
    print("   ✅ Agente creado")
    
    print("\n🔧 Paso 3/5: Reset y captura de 5 frames...")
    state, _ = env.reset()
    
    frames_captured = 0
    for i in range(5):
        # Renderizar (guardar imagen)
        env.render()
        frames_captured += 1
        
        # Acción aleatoria
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
        
        time.sleep(0.5)  # Pequeña pausa entre frames
        
        if done:
            print(f"   ℹ️  Episodio terminado en step {i+1}")
            break
    
    print(f"   ✅ {frames_captured} frames capturados")
    
    print("\n🔧 Paso 4/5: Analizando imágenes generadas...")
    
    # Verificar que se generaron archivos
    import glob
    frames = sorted(glob.glob('render_output/frame_*.png'))
    
    if not frames:
        print("   ❌ No se generaron imágenes en render_output/")
        print("   💡 Verificar que env.render() está activado")
    else:
        print(f"   ✅ {len(frames)} imágenes generadas")
        
        # Analizar primera imagen
        import cv2
        img = cv2.imread(frames[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extraer región del agente (330×330)
        agent_region = gray[0:330, 0:330]
        
        # Convertir a matriz 11×11
        matrix = np.zeros((11, 11), dtype=np.uint8)
        for i in range(11):
            for j in range(11):
                block = agent_region[i*30:(i+1)*30, j*30:(j+1)*30]
                matrix[i, j] = int(block.mean())
        
        print(f"\n   📊 Matriz 11×11 del agente:")
        print("   " + "-" * 58)
        for i, row in enumerate(matrix):
            row_str = " ".join(f"{val:3d}" for val in row)
            print(f"   {i:2d}: {row_str}")
        
        # Estadísticas
        print(f"\n   📈 Estadísticas:")
        print(f"      Min: {matrix.min()}")
        print(f"      Max: {matrix.max()}")
        print(f"      Mean: {matrix.mean():.1f}")
        print(f"      Std: {matrix.std():.1f}")
        
        # Variación horizontal
        h_vars = [np.std(row) for row in matrix]
        avg_h_var = np.mean(h_vars)
        
        print(f"\n   🔍 Variación horizontal promedio: {avg_h_var:.2f}")
        
        # Evaluación
        if avg_h_var < 5:
            print("      ❌ MUY BAJA - Todavía parece líneas horizontales")
            print("      💡 Puede necesitar más ajuste de cámara")
        elif avg_h_var < 15:
            print("      ⚠️  BAJA - Algo de variación pero mejorable")
        else:
            print("      ✅ BUENA - Imagen tiene variación adecuada")
        
        # Filas uniformes
        uniform_rows = 0
        for i in range(10):
            diff = np.abs(matrix[i].astype(int) - matrix[i+1].astype(int)).mean()
            if diff < 3:
                uniform_rows += 1
        
        print(f"\n   📏 Filas consecutivas casi idénticas: {uniform_rows}/10")
        
        if uniform_rows > 7:
            print("      ❌ Muchas filas uniformes")
        elif uniform_rows > 4:
            print("      ⚠️  Algunas filas uniformes")
        else:
            print("      ✅ Buena variación entre filas")
    
    print("\n🔧 Paso 5/5: Limpiando...")
    env.close()
    print("   ✅ Entorno cerrado")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)
    
    # Resumen final
    print("\n📝 RESUMEN:")
    if frames:
        print(f"   ✅ Configuración de cámara: CORRECTA (pitch=-15°)")
        print(f"   ✅ Render: ACTIVADO (guardando en render_output/)")
        print(f"   ✅ Frames generados: {len(frames)}")
        
        if avg_h_var >= 15 and uniform_rows <= 4:
            print(f"   ✅ Calidad de imagen: EXCELENTE")
            print(f"\n🎉 ¡Todo funciona perfectamente!")
            print(f"   La cámara ahora captura la carretera correctamente")
            print(f"   El agente tiene buena información para aprender")
        elif avg_h_var >= 10:
            print(f"   ⚠️  Calidad de imagen: ACEPTABLE")
            print(f"   La imagen tiene variación pero podría ser mejor")
        else:
            print(f"   ❌ Calidad de imagen: PROBLEMA")
            print(f"   Todavía se ven principalmente líneas horizontales")
        
        print(f"\n📁 Ver imágenes en: render_output/")
        print(f"   - frame_0000.png (primera captura)")
        print(f"   - frame_0001.png, frame_0002.png, etc.")
    else:
        print("   ❌ No se generaron imágenes")
        print("   💡 Verificar que env.render() está activado en src/main.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

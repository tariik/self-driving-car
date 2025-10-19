#!/usr/bin/env python3
"""
ANÁLISIS PASO A PASO: Flujo completo de imágenes
Desde sensor CARLA → procesamiento → agente → render
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import numpy as np
from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv

config = {
    "carla": {"host": "localhost", "port": 3000, "timeout": 60.0, "timestep": 0.1, "use_external_server": True, "enable_rendering": True},
    "others": {"framestack": 1, "max_time_idle": 100, "max_time_episode": 500}
}

experiment_config = BASE_EXPERIMENT_CONFIG.copy()
experiment_config["use_random_routes"] = True
experiment_config["others"] = config["others"]  # ← Agregar antes de BaseEnv
experiment_config["hero"]["sensors"] = {
    "rgb_camera": {
        "type": "sensor.camera.rgb",
        "transform": "2.0,0.0,1.2,0.0,-15.0,0.0",
        "image_size_x": "640",
        "image_size_y": "480",
        "fov": "90",
        "size": 11
    }
}

debug_dir = 'flujo_debug'
os.makedirs(debug_dir, exist_ok=True)

print("="*80)
print("🔍 FLUJO COMPLETO DE IMÁGENES: Sensor → Agente → Render")
print("="*80)

experiment = BaseEnv(experiment_config)
# Ya no necesita esta línea porque lo agregamos arriba
# experiment.config["others"] = config["others"]

# Interceptar post_process_image
original_pp = experiment.post_process_image
frame_num = [0]

def debug_pp(sensor_data, normalized=True, grayscale=True):
    n = frame_num[0]
    raw = sensor_data['rgb_camera'][1]
    if isinstance(raw, list):
        raw = raw[0]
    
    print(f"\n{'─'*80}")
    print(f"FRAME {n}")
    print(f"{'─'*80}")
    
    print(f"\n📷 PASO 1: Sensor CARLA (RAW 640×480 RGB)")
    print(f"   Shape: {raw.shape}, Min: {raw.min()}, Max: {raw.max()}, Mean: {raw.mean():.1f}")
    cv2.imwrite(f'{debug_dir}/f{n}_1_raw_640x480.png', cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    
    # Análisis horizontal RAW
    gray_raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    h_var_raw = np.mean([np.std(row) for row in gray_raw])
    print(f"   H-Var: {h_var_raw:.1f} {'❌ MUY UNIFORME' if h_var_raw < 20 else '✅ OK'}")
    
    print(f"\n🎨 PASO 2: RGB → Grayscale (640×480)")
    gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    print(f"   Shape: {gray.shape}")
    cv2.imwrite(f'{debug_dir}/f{n}_2_grayscale_640x480.png', gray)
    
    print(f"\n📐 PASO 3: Resize → 11×11")
    resized = cv2.resize(gray, (11, 11))
    print(f"   Shape: {resized.shape}")
    
    print(f"\n   Matriz 11×11:")
    for i, row in enumerate(resized):
        vals = " ".join(f"{v:3d}" for v in row)
        std_row = np.std(row)
        status = "←❌ UNIFORME" if std_row < 5 else ""
        print(f"   {i:2d}: {vals} {status}")
    
    h_var_11 = np.mean([np.std(row) for row in resized])
    uniform_rows = sum(1 for row in resized if np.std(row) < 5)
    print(f"\n   H-Var: {h_var_11:.1f}")
    print(f"   Filas uniformes: {uniform_rows}/11")
    print(f"   {'❌ LÍNEAS HORIZONTALES' if h_var_11 < 10 else '✅ BUENA VARIACIÓN'}")
    
    cv2.imwrite(f'{debug_dir}/f{n}_3_resized_11x11_x30.png', cv2.resize(resized, (330,330), interpolation=cv2.INTER_NEAREST))
    
    print(f"\n🔢 PASO 4: Normalizar → [-1, 1]")
    if len(resized.shape) == 2:
        resized = resized[:, :, np.newaxis]
    
    if normalized:
        norm = (resized.astype(np.float32) - 128) / 128
        print(f"   Min: {norm.min():.3f}, Max: {norm.max():.3f}")
        result = norm
    else:
        result = resized.astype(np.uint8)
    
    frame_num[0] += 1
    return result

experiment.post_process_image = debug_pp

try:
    env = CarlaEnv(experiment, config)
    
    print("\n🔄 Reset...")
    state, _ = env.reset()
    
    print(f"\n{'='*80}")
    print(f"📦 PASO 5: Estado del AGENTE")
    print(f"{'='*80}")
    print(f"   Shape: {state.shape}")
    print(f"   [0:121]  = Imagen 11×11 aplanada")
    print(f"   [121]    = φt = {state[121]:.6f} rad ({np.degrees(state[121]):.2f}°)")
    print(f"   [122]    = dt = {state[122]:.6f} m")
    
    print(f"\n🎬 Tomando 2 pasos...")
    for i in range(2):
        state, _, done, _, _ = env.step(13)
        env.render()
        if done:
            break
    
    print(f"\n{'='*80}")
    print(f"💾 PASO 6: RENDER guardado")
    print(f"{'='*80}")
    print(f"   El render toma:")
    print(f"      state[0:121] → reshape(11,11) → desnorm → scale(330,330)")
    print(f"   Guardado en: render_output/frame_*.png")
    
    print(f"\n{'='*80}")
    print(f"✅ ANÁLISIS COMPLETADO")
    print(f"{'='*80}")
    print(f"\n📁 Archivos:")
    print(f"   {debug_dir}/f*_1_raw_640x480.png      - RAW del sensor")
    print(f"   {debug_dir}/f*_2_grayscale_640x480.png - Grayscale")
    print(f"   {debug_dir}/f*_3_resized_11x11_x30.png - 11×11 escalado")
    print(f"   render_output/frame_*.png              - Render final")
    
    print(f"\n🔍 DIAGNÓSTICO:")
    print(f"   Si H-Var RAW < 20  → Imagen del sensor ya es uniforme")
    print(f"   Si H-Var 11×11 < 10 → Hay líneas horizontales")
    
    env.close()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

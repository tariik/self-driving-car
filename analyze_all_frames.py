#!/usr/bin/env python3
"""Analizar TODOS los frames para encontrar patrones"""
from PIL import Image
import numpy as np
import glob

frames = sorted(glob.glob('render_output/frame_*.png'))
print(f"📁 Analizando {len(frames)} frames\n")
print("Frame | H-Var | Filas Unif | Estado")
print("-" * 50)

for frame_path in frames:
    frame_num = int(frame_path.split('_')[-1].split('.')[0])
    
    img = Image.open(frame_path)
    arr = np.array(img.convert('L'))
    
    # Extraer matriz 11x11 (downsampling de 330x330)
    sample = arr[::30, ::30][:11, :11]
    
    # Métricas
    h_var = np.mean([np.std(row) for row in sample])
    uniform_rows = sum(1 for row in sample if np.std(row) < 5)
    
    # Estado
    if h_var < 5:
        estado = "❌ MUY MAL"
    elif h_var < 10:
        estado = "⚠️  MAL"
    elif h_var < 20:
        estado = "⚙️  REGULAR"
    else:
        estado = "✅ BIEN"
    
    print(f"{frame_num:04d}  | {h_var:5.1f} | {uniform_rows:2d}/11     | {estado}")

print("\n📊 Resumen:")
print("  H-Var < 10  = Líneas horizontales (cámara mal)")
print("  H-Var > 20  = Buena variación (cámara bien)")

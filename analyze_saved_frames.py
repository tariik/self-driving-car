#!/usr/bin/env python3
"""Analizar los frames guardados en render_output/"""
from PIL import Image
import numpy as np
import glob

frames = sorted(glob.glob('render_output/frame_*.png'))
if not frames:
    print("❌ No hay frames en render_output/")
    exit(1)

print(f"📁 Encontrados {len(frames)} frames")
print(f"📷 Analizando: {frames[5] if len(frames) > 5 else frames[0]}")

# Cargar imagen
img = Image.open(frames[5] if len(frames) > 5 else frames[0])
print(f"   Tamaño: {img.size}")

# El render guarda la imagen escalada a 330x330
# Necesitamos extraer los valores originales de la imagen 11x11
# La función render() en carla_env.py escala con NEAREST, así que podemos
# hacer downsampling
gray = np.array(img.convert('L'))
# Downsampling 330x330 -> 11x11 (cada pixel del agente = 30x30 en render)
matrix_11x11 = gray[::30, ::30][:11, :11]

print(f"\n📊 Matriz 11x11 del agente:")
print("   " + "-" * 33)
for i, row in enumerate(matrix_11x11):
    print(f"{i:2d}: " + " ".join(f"{v:3d}" for v in row))

# Estadísticas
print(f"\n📈 Estadísticas:")
print(f"   Min: {matrix_11x11.min()}")
print(f"   Max: {matrix_11x11.max()}")
print(f"   Mean: {matrix_11x11.mean():.1f}")
print(f"   Std: {matrix_11x11.std():.1f}")

# Calcular variación horizontal (dentro de cada fila)
h_var = np.mean([np.std(row) for row in matrix_11x11])
print(f"\n🔍 Variación horizontal promedio: {h_var:.1f}")
if h_var < 10:
    print("   ❌ PROBLEMA: Líneas horizontales detectadas")
    print("   💡 La cámara probablemente NO está apuntando a la carretera")
else:
    print("   ✅ OK: Buena variación en la imagen")

# Contar filas uniformes
uniform_rows = sum(1 for row in matrix_11x11 if np.std(row) < 5)
print(f"\n📏 Filas casi uniformes: {uniform_rows}/11")
if uniform_rows > 5:
    print("   ❌ Demasiadas filas uniformes (problema de cámara)")
else:
    print("   ✅ Variación adecuada")

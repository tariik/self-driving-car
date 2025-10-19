#!/usr/bin/env python3
"""
Visualización mejorada: Muestra la matriz 11×11 con valores numéricos
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

print("=" * 70)
print("🔍 VISUALIZACIÓN DE MATRIZ 11×11 DEL AGENTE")
print("=" * 70)

# Leer imagen guardada
img = cv2.imread('render_output/frame_0000.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Extraer región central (sin texto)
# La imagen real del agente está en el centro
center_y = gray.shape[0] // 2
center_x = gray.shape[1] // 2
margin = 165  # 330/2 = 165

# Extraer región de 330x330 que corresponde a la imagen del agente
agent_region = gray[0:330, 0:330]

# Reducir a 11×11 (promediando cada bloque de 30×30)
matrix_11x11 = np.zeros((11, 11), dtype=np.uint8)
block_size = 30

for i in range(11):
    for j in range(11):
        y_start = i * block_size
        y_end = (i + 1) * block_size
        x_start = j * block_size
        x_end = (j + 1) * block_size
        
        block = agent_region[y_start:y_end, x_start:x_end]
        matrix_11x11[i, j] = int(block.mean())

print("\n📊 Matriz 11×11 (valores 0-255):")
print("   Columnas: 0   1   2   3   4   5   6   7   8   9  10")
print("   " + "-" * 58)

for i, row in enumerate(matrix_11x11):
    print(f"   {i:2d}: ", end="")
    for val in row:
        print(f"{val:3d} ", end="")
    print()

print("\n🎨 Visualización con caracteres:")
print("   (█ = oscuro, ░ = claro)")
print()

chars = " ░▒▓█"
for i, row in enumerate(matrix_11x11):
    print(f"   {i:2d}: ", end="")
    for val in row:
        # Mapear 0-255 a 0-4
        char_idx = min(4, val // 51)
        print(chars[char_idx] * 2, end=" ")
    print()

# Estadísticas
print(f"\n📈 Estadísticas de la matriz 11×11:")
print(f"   Min: {matrix_11x11.min()}")
print(f"   Max: {matrix_11x11.max()}")
print(f"   Mean: {matrix_11x11.mean():.2f}")
print(f"   Std: {matrix_11x11.std():.2f}")

# Verificar si hay patrones
print(f"\n🔍 Análisis de patrones:")

# Contar filas similares
similar_rows = 0
for i in range(10):
    diff = np.abs(matrix_11x11[i].astype(int) - matrix_11x11[i+1].astype(int)).mean()
    if diff < 10:
        similar_rows += 1

print(f"   Filas consecutivas similares: {similar_rows}/10")

if similar_rows > 7:
    print("\n⚠️  PROBLEMA: Muchas filas similares (efecto 'líneas horizontales')")
    print("   Posibles causas:")
    print("   1. Cámara apuntando al cielo o suelo uniforme")
    print("   2. Imagen capturada en zona sin textura")
    print("   3. Normalización/procesamiento incorrecto")
else:
    print(f"\n✅ Imagen tiene suficiente variación entre filas")

# Variación horizontal vs vertical
var_horizontal = np.mean([np.std(row) for row in matrix_11x11])
var_vertical = np.mean([np.std(matrix_11x11[:, col]) for col in range(11)])

print(f"\n📊 Variación:")
print(f"   Horizontal (dentro de filas): {var_horizontal:.2f}")
print(f"   Vertical (dentro de columnas): {var_vertical:.2f}")

if var_horizontal < 5 and var_vertical < 5:
    print("\n❌ PROBLEMA: Muy poca variación en ambas direcciones")
    print("   La imagen es casi uniforme - verificar sensor")
elif var_horizontal < 10:
    print("\n⚠️  Poca variación horizontal (puede verse como líneas)")
else:
    print("\n✅ Variación normal")

print("\n" + "=" * 70)
print("💡 NOTA: Una imagen 11×11 naturalmente se ve muy pixelada")
print("   Esto es parte del diseño del paper DRL-Flatten-Image")
print("=" * 70)

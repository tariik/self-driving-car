#!/usr/bin/env python3
import cv2
import numpy as np

print('='*80)
print('📊 ANÁLISIS DEL FLUJO COMPLETO - Frame 0')
print('='*80)

# PASO 1: RAW
raw = cv2.imread('flujo_debug/f0_1_raw_640x480.png', cv2.IMREAD_GRAYSCALE)
h_var_raw = np.mean([np.std(row) for row in raw])
print(f'\n📷 PASO 1: Imagen RAW 640×480')
print(f'   Shape: {raw.shape}')
print(f'   Variación horizontal: {h_var_raw:.1f}')
if h_var_raw < 20:
    print(f'   ❌ RAW YA ES UNIFORME! (problema del sensor/ubicación)')
else:
    print(f'   ✅ RAW tiene buena variación')

# PASO 2: Grayscale (ya está en gris)
gray = cv2.imread('flujo_debug/f0_2_grayscale_640x480.png', cv2.IMREAD_GRAYSCALE)
h_var_gray = np.mean([np.std(row) for row in gray])
print(f'\n🎨 PASO 2: Grayscale 640×480')
print(f'   Variación horizontal: {h_var_gray:.1f}')

# PASO 3: Resized 11x11
resized = cv2.imread('flujo_debug/f0_3_resized_11x11_x30.png', cv2.IMREAD_GRAYSCALE)
# Downsampling de 330x330 a 11x11
resized_11 = resized[::30, ::30]
h_var_11 = np.mean([np.std(row) for row in resized_11])
uniform_rows = sum(1 for row in resized_11 if np.std(row) < 5)

print(f'\n📐 PASO 3: Resize 11×11')
print(f'   Matriz:')
for i, row in enumerate(resized_11):
    vals = ' '.join(f'{v:3d}' for v in row)
    std = np.std(row)
    status = ' ←❌ UNIFORME' if std < 5 else ''
    print(f'   {i:2d}: {vals}{status}')

print(f'\n   Variación horizontal: {h_var_11:.1f}')
print(f'   Filas uniformes: {uniform_rows}/11')
if h_var_11 < 10:
    print(f'   ❌ LÍNEAS HORIZONTALES!')
else:
    print(f'   ✅ Buena variación')

# Render
render = cv2.imread('render_output/frame_0000.png', cv2.IMREAD_GRAYSCALE)
if render is not None:
    render_11 = render[::30, ::30][:11,:11]
    h_var_render = np.mean([np.std(row) for row in render_11])
    print(f'\n💾 RENDER final:')
    print(f'   Variación horizontal: {h_var_render:.1f}')

print(f'\n{'='*80}')
print(f'🔍 CONCLUSIÓN:')
print(f'{'='*80}')
print(f'   RAW 640×480:  H-Var = {h_var_raw:.1f}  {'❌' if h_var_raw < 20 else '✅'}')
print(f'   11×11:        H-Var = {h_var_11:.1f}   {'❌' if h_var_11 < 10 else '✅'}')
print()
if h_var_raw < 20 and h_var_11 < 10:
    print('   💡 DIAGNÓSTICO: La imagen RAW del sensor ya viene uniforme.')
    print('      El problema NO es el procesamiento.')
    print('      El vehículo está viendo cielo/edificios/asfalto uniforme.')
    print()
    print('   🔧 SOLUCIONES:')
    print('      1. Cambiar ángulo de cámara (probar pitch=-20° o -25°)')
    print('      2. Mejorar spawn locations')
    print('      3. Verificar que el vehículo no spawn en lugares sin textura')
elif h_var_raw > 20 and h_var_11 < 10:
    print('   💡 DIAGNÓSTICO: El RAW es bueno pero se pierde en el resize.')
    print('      Problema en el algoritmo de reducción 640x480 → 11x11')
else:
    print('   ✅ Todo funciona correctamente')

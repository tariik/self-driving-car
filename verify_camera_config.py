#!/usr/bin/env python3
"""
Verificar la configuración actual de la cámara en src/main.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.env.base_env import BASE_EXPERIMENT_CONFIG
import importlib.util

print("=" * 70)
print("🔍 VERIFICACIÓN DE CONFIGURACIÓN DE CÁMARA")
print("=" * 70)

# Cargar main.py y extraer la configuración
spec = importlib.util.spec_from_file_location("main", "src/main.py")
if spec and spec.loader:
    main_module = importlib.util.module_from_spec(spec)
    
    # No ejecutar, solo parsear para ver la configuración
    print("\n✅ Módulo src/main.py encontrado")
    
    # Leer el archivo directamente
    with open('src/main.py', 'r') as f:
        content = f.read()
    
    print("\n📷 Buscando configuración de cámara...")
    
    # Buscar la línea de transform
    for line_num, line in enumerate(content.split('\n'), 1):
        if '"transform"' in line and 'rgb_camera' in content[max(0, content.find(line)-500):content.find(line)+100]:
            print(f"\n   Línea {line_num}: {line.strip()}")
            
            # Extraer valores
            if '-15' in line:
                print("   ✅ CORRECTO: pitch=-15° (apuntando a la carretera)")
            elif '0.0,0.0,0.0' in line or '0,0,0' in line:
                print("   ❌ PROBLEMA: pitch=0° (apuntando horizontal)")
                print("   💡 Necesitas cambiar a pitch=-15°")
            else:
                print(f"   ⚠️  Revisar: {line}")

# Verificar BASE_EXPERIMENT_CONFIG
print("\n📋 BASE_EXPERIMENT_CONFIG sensores:")
if 'sensors' in BASE_EXPERIMENT_CONFIG['hero']:
    sensors = BASE_EXPERIMENT_CONFIG['hero']['sensors']
    if sensors:
        print(f"   {sensors}")
    else:
        print("   ✅ Vacío (se sobreescribe en main.py)")

print("\n" + "=" * 70)
print("📝 INSTRUCCIONES:")
print("=" * 70)
print()
print("Si ves pitch=0°, necesitas editar src/main.py:")
print()
print('  CAMBIAR de:  "transform": "1.5,0.0,1.5,0.0,0.0,0.0"')
print('  CAMBIAR a:   "transform": "2.0,0.0,1.2,0.0,-15.0,0.0"')
print()
print("Esto hace que la cámara apunte 15° hacia abajo (a la carretera)")
print("en lugar de horizontal (al cielo)")
print()
print("=" * 70)

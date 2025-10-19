"""
Script de prueba para verificar la visualización de rutas
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv


def main():
    print("\n" + "="*80)
    print("🧪 TEST: Visualización de Ruta")
    print("="*80 + "\n")
    
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
    experiment_config["clean_road"] = True
    
    experiment_config["hero"]["sensors"] = {
        "rgb_camera": {
            "type": "sensor.camera.rgb",
            "transform": "2.0,0.0,1.2,0.0,-25.0,0.0",
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
    
    print("Creando entorno...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    print("✅ Entorno creado\n")
    
    try:
        # Test 1: Generar primera ruta
        print("🧪 TEST 1: Generando primera ruta...")
        state, _ = env.reset()
        print("✅ Primera ruta generada y visualizada\n")
        print("⏸️  Esperando 10 segundos para inspeccionar...")
        time.sleep(10)
        
        # Test 2: Generar segunda ruta (debería limpiar la anterior)
        print("\n🧪 TEST 2: Generando segunda ruta (la primera debería desaparecer)...")
        state, _ = env.reset()
        print("✅ Segunda ruta generada y visualizada\n")
        print("⏸️  Esperando 10 segundos para inspeccionar...")
        time.sleep(10)
        
        # Test 3: Generar tercera ruta
        print("\n🧪 TEST 3: Generando tercera ruta...")
        state, _ = env.reset()
        print("✅ Tercera ruta generada y visualizada\n")
        print("⏸️  Esperando 10 segundos para inspeccionar...")
        time.sleep(10)
        
        print("\n" + "="*80)
        print("✅ TEST COMPLETADO")
        print("="*80)
        print("\nVerifica en CARLA:")
        print("1. ¿Ves UN marcador azul (START)?")
        print("2. ¿Ves UNA línea verde conectando waypoints?")
        print("3. ¿Ves UN marcador rojo (GOAL)?")
        print("4. ¿Las rutas anteriores desaparecieron después de ~60 segundos?")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrumpido por usuario")
    finally:
        print("\nCerrando entorno...")
        env.close()
        print("✅ Limpieza completa\n")


if __name__ == "__main__":
    main()

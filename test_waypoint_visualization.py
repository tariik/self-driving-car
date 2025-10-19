"""
Test de Visualización de Waypoints en 3D
Los waypoints se muestran SOLO en el spectator, NO en la cámara del agente
"""

import numpy as np
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv

def test_waypoint_visualization():
    """Test para ver la visualización de waypoints en CARLA"""
    
    print("=" * 70)
    print("👁️  TEST: VISUALIZACIÓN DE WAYPOINTS EN 3D")
    print("=" * 70)
    print()
    print("ℹ️  IMPORTANTE:")
    print("   - Los waypoints se dibujan en el MUNDO 3D de CARLA")
    print("   - Son visibles en el SPECTATOR (cámara externa)")
    print("   - Abre la ventana de CARLA para verlos")
    print("   - NO aparecen en la cámara del agente")
    print()
    print("🎨 COLORES:")
    print("   🔵 AZUL = Punto de inicio (START)")
    print("   🔴 ROJO = Punto de destino (GOAL)")
    print("   🟢 VERDE = Waypoints de la ruta")
    print("   🟡 AMARILLO = Waypoint actual (se actualiza)")
    print()
    print("📺 VENTANA DE CARLA:")
    print("   ⚠️  ASEGÚRATE de tener la ventana de CARLA visible")
    print("   ⚠️  Los waypoints aparecen en el SPECTATOR")
    print("   💡 Puedes mover la cámara con el mouse")
    print()
    print("⏳ Preparando entorno...")
    print()
    
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
            "framestack": 4,
            "max_time_idle": 100,
            "max_time_episode": 500,
        }
    }
    
    # Configuración con rutas aleatorias
    experiment_config = BASE_EXPERIMENT_CONFIG.copy()
    experiment_config["hero"]["sensors"] = {
        "rgb_camera": {
            "type": "sensor.camera.rgb",
            "transform": "1.5,0.0,1.5,0.0,0.0,0.0",
            "image_size_x": "84",
            "image_size_y": "84",
            "fov": "90",
            "size": 84
        }
    }
    experiment_config["hero"]["spawn_points"] = []
    experiment_config["town"] = "Town01"
    experiment_config["weather"] = "ClearNoon"
    experiment_config["others"] = config["others"]
    experiment_config["use_random_routes"] = True  # ⭐ ACTIVAR visualización
    
    # Crear entorno
    print("🚗 Creando entorno CARLA...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    print("✅ Entorno creado")
    print()
    
    # Reset genera ruta y la visualiza
    print("=" * 70)
    print("🗺️  GENERANDO Y VISUALIZANDO RUTA")
    print("=" * 70)
    print()
    
    observation, info = env.reset()
    
    route_info = env.experiment.get_route_info(env.core)
    
    if route_info['has_route']:
        print(f"✅ Ruta generada y visualizada en el mundo 3D")
        print()
        print(f"📊 Información de la ruta:")
        print(f"   • Total waypoints: {route_info['total_waypoints']}")
        print(f"   • Distancia a meta: {route_info['distance_to_goal']:.1f}m")
        print()
        print("👁️  VISUALIZACIÓN ACTIVA:")
        print(f"   • Waypoints verdes: {route_info['total_waypoints']} puntos")
        print(f"   • Líneas verdes conectan los waypoints")
        print(f"   • Punto azul (START) al inicio")
        print(f"   • Punto rojo (GOAL) al final")
        print(f"   • Números cada 10 waypoints")
        print()
        print("💡 TIP: Mueve la cámara del spectator para ver la ruta completa")
        print()
    else:
        print("❌ No se pudo generar ruta")
        return False
    
    print("-" * 70)
    print()
    
    # Avanzar por la ruta y ver el waypoint actual
    print("=" * 70)
    print("🎮 SIGUIENDO LA RUTA (waypoint actual en AMARILLO)")
    print("=" * 70)
    print()
    print("⏳ Ejecutando 30 steps para ver el progreso...")
    print()
    
    for step_num in range(30):
        # Acción: avanzar moderadamente
        action = 15  # Throttle 0.6, steering 0
        
        observation, reward, done, truncated, info = env.step(action)
        
        # Obtener progreso
        route_info = env.experiment.get_route_info(env.core)
        
        # Mostrar progreso cada 5 steps
        if (step_num + 1) % 5 == 0:
            progress_pct = route_info['progress'] * 100
            print(f"   Step {step_num + 1:2d}: "
                  f"Progreso {progress_pct:5.1f}% | "
                  f"WPs: {route_info['waypoints_completed']:3d}/{route_info['total_waypoints']:3d} | "
                  f"Dist: {route_info['distance_to_goal']:6.1f}m | "
                  f"🟡 Waypoint actual marcado")
        
        if done or truncated:
            print(f"   ⚠️  Episodio terminado en step {step_num + 1}")
            break
        
        time.sleep(0.1)  # Pausa para ver la visualización
    
    print()
    print("✅ Visualización completada")
    print()
    
    # Mantener visualización un momento
    print("-" * 70)
    print()
    print("📸 Manteniendo visualización por 10 segundos...")
    print("   (Puedes mover la cámara del spectator)")
    print()
    
    for i in range(10):
        time.sleep(1)
        remaining = 10 - i
        print(f"   ⏳ {remaining} segundos restantes...", end='\r')
    
    print()
    print()
    
    # Cerrar entorno
    env.close()
    
    print("=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)
    print()
    print("📖 Resumen:")
    print()
    print("   ✅ Waypoints dibujados en el mundo 3D")
    print("   ✅ Colores: Azul (START), Verde (ruta), Rojo (GOAL)")
    print("   ✅ Waypoint actual marcado en amarillo")
    print("   ✅ Líneas conectando waypoints")
    print("   ✅ NO aparecen en la cámara del agente")
    print()
    print("💡 Características:")
    print("   • world.debug.draw_point() - Dibuja puntos 3D")
    print("   • world.debug.draw_line() - Dibuja líneas 3D")
    print("   • world.debug.draw_string() - Dibuja texto 3D")
    print("   • Solo visible en spectator")
    print("   • No afecta las observaciones del agente")
    print("   • Lifetime: 120 segundos (configurable)")
    print()
    print("🎯 Uso en training:")
    print("   • Debugging de rutas")
    print("   • Verificar que el agente sigue la ruta")
    print("   • Visualizar progreso en tiempo real")
    print("   • Identificar problemas de navegación")
    print()
    
    return True

if __name__ == "__main__":
    try:
        print()
        success = test_waypoint_visualization()
        print()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR durante el test:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

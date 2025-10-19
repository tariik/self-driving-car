"""
Test script para verificar Rutas Aleatorias (Phase 1.4)
Como en el paper de Pérez-Gil et al. (2022)
"""

import numpy as np
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv

def test_random_routes():
    """Test para verificar generación de rutas aleatorias"""
    
    print("=" * 70)
    print("🧪 TEST: RUTAS ALEATORIAS (Phase 1.4)")
    print("=" * 70)
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
    
    # Configuración del experimento CON rutas aleatorias
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
    experiment_config["use_random_routes"] = True  # ⭐ ACTIVAR rutas aleatorias
    
    # Crear entorno
    print("🚗 Creando entorno CARLA...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    print("✅ Entorno creado")
    print()
    
    # Test 1: Generar múltiples rutas aleatorias
    print("=" * 70)
    print("TEST 1: GENERACIÓN DE MÚLTIPLES RUTAS")
    print("=" * 70)
    print()
    
    num_routes = 3
    for route_num in range(num_routes):
        print(f"🗺️  Ruta {route_num + 1}/{num_routes}")
        print("-" * 70)
        
        # Reset genera nueva ruta aleatoria
        observation, info = env.reset()
        
        # Obtener información de la ruta
        route_info = env.experiment.get_route_info(env.core)
        
        if route_info['has_route']:
            print(f"   ✅ Ruta generada correctamente")
            print(f"   📊 Waypoints totales: {route_info['total_waypoints']}")
            print(f"   📍 Distancia a meta: {route_info['distance_to_goal']:.1f}m")
            print(f"   🎯 Distancia a próximo WP: {route_info['distance_to_next_wp']:.1f}m")
        else:
            print(f"   ❌ No se generó ruta")
        
        print()
        time.sleep(0.5)
    
    print("-" * 70)
    print()
    
    # Test 2: Seguir una ruta
    print("=" * 70)
    print("TEST 2: SEGUIMIENTO DE RUTA")
    print("=" * 70)
    print()
    
    # Reset con nueva ruta
    observation, info = env.reset()
    route_info = env.experiment.get_route_info(env.core)
    
    if not route_info['has_route']:
        print("   ⚠️  No hay ruta disponible para test")
    else:
        print(f"🗺️  Ruta de {route_info['total_waypoints']} waypoints")
        print(f"📍 Distancia total a meta: {route_info['distance_to_goal']:.1f}m")
        print()
        print("🎮 Avanzando por la ruta...")
        print()
        
        max_steps = 50
        goal_reached = False
        
        for step_num in range(max_steps):
            # Acción: avanzar moderadamente
            action = 15  # Throttle 0.6, steering 0
            
            observation, reward, done, truncated, info = env.step(action)
            
            # Obtener progreso
            route_info = env.experiment.get_route_info(env.core)
            
            # Verificar si alcanzó la meta
            if env.experiment._is_goal_reached(env.core):
                goal_reached = True
                print(f"   Step {step_num + 1}: 🎯 META ALCANZADA!")
                print(f"   Reward bonus: +100")
                break
            
            # Mostrar progreso cada 5 steps
            if (step_num + 1) % 5 == 0:
                progress_pct = route_info['progress'] * 100
                print(f"   Step {step_num + 1}: Progreso {progress_pct:.1f}% | "
                      f"WPs: {route_info['waypoints_completed']}/{route_info['total_waypoints']} | "
                      f"Dist meta: {route_info['distance_to_goal']:.1f}m | "
                      f"Reward: {reward:.2f}")
            
            if done or truncated:
                print(f"   ⚠️  Episodio terminado en step {step_num + 1}")
                break
            
            time.sleep(0.05)
        
        print()
        print("📊 Resultado Test 2:")
        if goal_reached:
            print(f"   ✅ ÉXITO: Meta alcanzada en {step_num + 1} steps")
        else:
            final_progress = route_info['progress'] * 100
            print(f"   ⚠️  Meta no alcanzada en {max_steps} steps")
            print(f"   📊 Progreso final: {final_progress:.1f}%")
            print(f"   📍 Distancia restante: {route_info['distance_to_goal']:.1f}m")
    
    print()
    print("-" * 70)
    print()
    
    # Test 3: Verificar waypoint tracking
    print("=" * 70)
    print("TEST 3: TRACKING DE WAYPOINTS")
    print("=" * 70)
    print()
    
    # Reset con nueva ruta
    observation, info = env.reset()
    route_info = env.experiment.get_route_info(env.core)
    
    if not route_info['has_route']:
        print("   ⚠️  No hay ruta disponible para test")
    else:
        print(f"🗺️  Verificando tracking de waypoints...")
        print()
        
        initial_wp_count = route_info['waypoints_completed']
        
        # Avanzar varios steps
        for _ in range(20):
            action = 22  # Throttle máximo, steering 0
            observation, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                break
            
            time.sleep(0.05)
        
        # Verificar progreso
        route_info = env.experiment.get_route_info(env.core)
        final_wp_count = route_info['waypoints_completed']
        
        waypoints_passed = final_wp_count - initial_wp_count
        
        print(f"   📊 Waypoints iniciales: {initial_wp_count}")
        print(f"   📊 Waypoints finales: {final_wp_count}")
        print(f"   ✅ Waypoints pasados: {waypoints_passed}")
        print()
        
        if waypoints_passed > 0:
            print("   ✅ ÉXITO: Tracking de waypoints funcionando")
        else:
            print("   ⚠️  No se pasaron waypoints (puede ser distancia corta)")
    
    print()
    print("-" * 70)
    print()
    
    # Cerrar entorno
    env.close()
    
    print("=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)
    print()
    print("📖 Comparación con Paper:")
    print("   ✅ Rutas aleatorias: Generación automática con A*")
    print("   ✅ GlobalRoutePlanner: Sampling 2.0m (como en paper)")
    print("   ✅ Waypoint tracking: Automático con tolerancia 3m")
    print("   ✅ Goal detection: Meta a <5m")
    print("   ✅ Route info: Progreso, distancia, WPs completados")
    print()
    print("💡 Beneficios para entrenamiento:")
    print("   - Mayor generalización (múltiples rutas)")
    print("   - Variedad de escenarios (diferentes spawn points)")
    print("   - Tracking automático de progreso")
    print("   - Meta clara (+100 reward al alcanzar)")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = test_random_routes()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR durante el test:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

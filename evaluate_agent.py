"""
Evaluación de Agente DRL-Flatten-Image Entrenado
=================================================

Basado en el paper de Pérez-Gil et al. 2022 - Sección 6.1.2 (Validation Stage)

Este script evalúa un modelo entrenado en una ruta específica,
calculando métricas de rendimiento (RMSE, error máximo, tiempo).
"""

import os
import sys
import numpy as np
import time
import glob

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agents.drl_flatten_agent import DRLFlattenAgent
from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv
from src.utils.video_recorder import VideoRecorder
from src.utils.display_manager import DisplayManager


def calculate_rmse(trajectory, reference_waypoints):
    """
    Calcula el RMSE entre la trayectoria conducida y los waypoints de referencia.
    
    Params
    ======
        trajectory: Lista de posiciones [(x, y, z), ...] del vehículo
        reference_waypoints: Lista de waypoints de CARLA de la ruta
    
    Returns
    =======
        rmse (float): Root Mean Square Error en metros
        max_error (float): Error máximo en metros
    """
    if not trajectory or not reference_waypoints:
        return float('inf'), float('inf')
    
    errors = []
    for pos in trajectory:
        # Encontrar el waypoint más cercano
        min_distance = float('inf')
        for wp in reference_waypoints:
            wp_loc = wp.transform.location
            distance = np.sqrt(
                (pos[0] - wp_loc.x)**2 + 
                (pos[1] - wp_loc.y)**2
            )
            if distance < min_distance:
                min_distance = distance
        errors.append(min_distance)
    
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    max_error = np.max(errors)
    
    return rmse, max_error


def evaluate_agent(checkpoint_path, num_iterations=20, save_video=True, show_display=True):
    """
    Evalúa un agente entrenado en una ruta específica.
    
    Paper (Tabla 3): Cada agente conduce la ruta 20 veces para calcular métricas.
    
    Params
    ======
        checkpoint_path (str): Ruta al checkpoint del modelo entrenado
        num_iterations (int): Número de veces que se conduce la ruta (default: 20)
        save_video (bool): Si True, guarda video de la evaluación
        show_display (bool): Si True, muestra ventana de visualización
    
    Returns
    =======
        results (dict): Diccionario con métricas de evaluación
    """
    
    print("\n" + "="*80)
    print("🎯 EVALUACIÓN DE AGENTE DRL-FLATTEN-IMAGE")
    print("="*80)
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"🔄 Iteraciones: {num_iterations}")
    print("="*80 + "\n")
    
    # 1. Configuración del entorno (igual que en training)
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
            "max_time_episode": 1000,  # Más tiempo para evaluación
        }
    }
    
    experiment_config = BASE_EXPERIMENT_CONFIG.copy()
    experiment_config["use_random_routes"] = True  # Rutas aleatorias
    
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
    experiment_config["town"] = "Town01"  # Paper usa Town01
    experiment_config["weather"] = "ClearNoon"
    experiment_config["others"] = config["others"]
    
    # 2. Crear entorno y agente
    print("🏗️  Creando entorno...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    print("🤖 Creando agente...")
    agent = DRLFlattenAgent(env, seed=0)
    
    # 3. Cargar modelo entrenado
    print(f"📦 Cargando modelo desde {checkpoint_path}...")
    agent.load(checkpoint_path)
    
    # 4. Configurar visualización
    display = None
    if show_display:
        try:
            display = DisplayManager(
                world=env.core.world,
                hero_vehicle=env.hero,
                width=800,
                height=600,
                follow_spectator=True
            )
            print("✅ Display activado")
        except Exception as e:
            print(f"⚠️  No se pudo abrir display: {e}")
            show_display = False
    
    # 5. Configurar grabación de video
    recorder = None
    if save_video:
        recorder = VideoRecorder(output_dir='evaluation_output', fps=10)
        print("✅ Grabación de video activada")
    
    # 6. Métricas de evaluación
    results = {
        'rmse_values': [],
        'max_errors': [],
        'times': [],
        'rewards': [],
        'trajectories': [],
        'route_lengths': [],
        'successes': [],
        'collision_count': 0,
        'lane_invasion_count': 0
    }
    
    # 7. Ejecutar evaluaciones
    print("\n🚗 Iniciando evaluaciones...\n")
    
    for iteration in range(num_iterations):
        print(f"{'='*80}")
        print(f"🔄 Iteración {iteration + 1}/{num_iterations}")
        print(f"{'='*80}")
        
        # Reset para nueva ruta
        state, _ = env.reset()
        
        # Guardar waypoints de referencia
        reference_waypoints = env.experiment.route_waypoints.copy()
        route_length = len(reference_waypoints)
        
        # Variables de la iteración
        trajectory = []
        total_reward = 0.0
        step = 0
        start_time = time.time()
        success = False
        
        # Ejecutar episodio COMPLETO (sin epsilon-greedy, solo explotación)
        while step < config["others"]["max_time_episode"]:
            # User quit check
            if display and display.process_events():
                print("⚠️  Usuario solicitó salir")
                break
            
            # Agente selecciona acción SIN exploración (eps=0)
            action = agent.act(state, eps=0.0)  # Modo evaluación: sin exploración
            
            # Guardar posición actual para trayectoria
            hero_loc = env.hero.get_location()
            trajectory.append((hero_loc.x, hero_loc.y, hero_loc.z))
            
            # Ejecutar acción
            next_state, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            step += 1
            
            # Actualizar visualización
            if display:
                display.update(
                    step=step,
                    reward=reward,
                    total_reward=total_reward,
                    done=done
                )
            
            # Grabar frame
            if recorder:
                recorder.add_frame(
                    observation=state,
                    step=step,
                    reward=reward,
                    total_reward=total_reward,
                    done=done
                )
            
            # Verificar terminación
            if done:
                # Verificar si llegó a la meta
                if env.experiment._is_goal_reached(env.core):
                    success = True
                    print(f"   ✅ Meta alcanzada!")
                elif env.experiment.collision_triggered:
                    results['collision_count'] += 1
                    print(f"   💥 Colisión detectada")
                elif env.experiment.lane_invasion_triggered:
                    results['lane_invasion_count'] += 1
                    print(f"   🚧 Invasión de carril")
                break
        
        elapsed_time = time.time() - start_time
        
        # Calcular RMSE para esta iteración
        rmse, max_error = calculate_rmse(trajectory, reference_waypoints)
        
        # Guardar resultados
        results['rmse_values'].append(rmse)
        results['max_errors'].append(max_error)
        results['times'].append(elapsed_time)
        results['rewards'].append(total_reward)
        results['trajectories'].append(trajectory)
        results['route_lengths'].append(route_length)
        results['successes'].append(success)
        
        # Imprimir resumen de la iteración
        print(f"   📊 Steps: {step}")
        print(f"   🎁 Reward: {total_reward:.2f}")
        print(f"   📏 RMSE: {rmse:.4f} m")
        print(f"   ❌ Max Error: {max_error:.4f} m")
        print(f"   ⏱️  Time: {elapsed_time:.2f} s")
        print(f"   🎯 Success: {'✅' if success else '❌'}")
        print()
    
    # 8. Cerrar recursos
    if display:
        display.close()
    if recorder:
        recorder.close()
    env.close()
    
    # 9. Calcular estadísticas finales
    print("\n" + "="*80)
    print("📈 RESULTADOS DE EVALUACIÓN")
    print("="*80)
    
    avg_rmse = np.mean(results['rmse_values'])
    std_rmse = np.std(results['rmse_values'])
    avg_max_error = np.mean(results['max_errors'])
    avg_time = np.mean(results['times'])
    success_rate = sum(results['successes']) / num_iterations * 100
    
    print(f"\n🎯 Métricas de Precisión:")
    print(f"   RMSE promedio: {avg_rmse:.4f} ± {std_rmse:.4f} m")
    print(f"   Error máximo promedio: {avg_max_error:.4f} m")
    print(f"   Mejor RMSE: {min(results['rmse_values']):.4f} m")
    print(f"   Peor RMSE: {max(results['rmse_values']):.4f} m")
    
    print(f"\n⏱️  Métricas de Tiempo:")
    print(f"   Tiempo promedio: {avg_time:.2f} s")
    
    print(f"\n🎁 Métricas de Recompensa:")
    print(f"   Reward promedio: {np.mean(results['rewards']):.2f}")
    print(f"   Reward total: {sum(results['rewards']):.2f}")
    
    print(f"\n✅ Métricas de Éxito:")
    print(f"   Tasa de éxito: {success_rate:.1f}% ({sum(results['successes'])}/{num_iterations})")
    print(f"   Colisiones: {results['collision_count']}")
    print(f"   Invasiones de carril: {results['lane_invasion_count']}")
    
    print(f"\n📊 Comparación con Paper (Tabla 3):")
    print(f"   LQR Controller: RMSE = 0.06 m (baseline)")
    print(f"   DQN-Flatten-Image: RMSE = 0.08 m (paper)")
    print(f"   DDPG-Flatten-Image: RMSE = 0.07 m (paper)")
    print(f"   Este modelo: RMSE = {avg_rmse:.4f} m")
    
    if avg_rmse < 0.10:
        print(f"   🏆 ¡Excelente! RMSE < 0.10 m")
    elif avg_rmse < 0.15:
        print(f"   ✅ Bueno. RMSE < 0.15 m")
    else:
        print(f"   ⚠️  Necesita más entrenamiento")
    
    print("="*80 + "\n")
    
    # 10. Guardar resultados en archivo
    results_file = f"evaluation_results_{os.path.basename(checkpoint_path).replace('.pth', '')}.txt"
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Iterations: {num_iterations}\n")
        f.write(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f} m\n")
        f.write(f"Max Error: {avg_max_error:.4f} m\n")
        f.write(f"Avg Time: {avg_time:.2f} s\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
        f.write(f"Collisions: {results['collision_count']}\n")
        f.write(f"Lane Invasions: {results['lane_invasion_count']}\n")
    
    print(f"💾 Resultados guardados en: {results_file}")
    
    return results


def find_latest_checkpoint():
    """Encuentra el checkpoint más reciente en el directorio checkpoints/"""
    checkpoints = glob.glob('checkpoints/drl_flatten_*.pth')
    if not checkpoints:
        return None
    
    # Ordenar por fecha de modificación
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def main():
    """Main function para evaluación"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar agente DRL-Flatten-Image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Ruta al checkpoint (default: último checkpoint)')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Número de iteraciones (default: 20, como en paper)')
    parser.add_argument('--no-video', action='store_true',
                        help='No guardar video de evaluación')
    parser.add_argument('--no-display', action='store_true',
                        help='No mostrar ventana de visualización')
    
    args = parser.parse_args()
    
    # Buscar checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No se encontraron checkpoints en 'checkpoints/'")
            print("   Entrena primero un modelo con: python src/main.py")
            return
        print(f"📦 Usando último checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return
    
    # Evaluar
    results = evaluate_agent(
        checkpoint_path=checkpoint_path,
        num_iterations=args.iterations,
        save_video=not args.no_video,
        show_display=not args.no_display
    )
    
    print("\n✅ Evaluación completada!")


if __name__ == "__main__":
    main()

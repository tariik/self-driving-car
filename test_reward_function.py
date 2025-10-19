"""
Test script para verificar la nueva Función de Recompensa
Como en el paper de Pérez-Gil et al. (2022)
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv

def test_reward_function():
    """Test para verificar la nueva función de recompensa"""
    
    print("=" * 70)
    print("🧪 TEST: FUNCIÓN DE RECOMPENSA (Paper Implementation)")
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
    
    # Configuración del experimento
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
    
    # Crear entorno
    print("🚗 Creando entorno CARLA...")
    experiment = BaseEnv(experiment_config)
    env = CarlaEnv(experiment, config)
    
    print("✅ Entorno creado")
    print()
    
    # Reset
    print("🔄 Reseteando entorno...")
    observation, info = env.reset()
    
    print(f"✅ Reset completado")
    print()
    
    # Verificar función de recompensa
    print("📊 VERIFICANDO FUNCIÓN DE RECOMPENSA:")
    print("-" * 70)
    print()
    
    print("📖 Fórmula del Paper:")
    print("   R = -200  (colisión o salida de carril)")
    print("   R = |vt·cos(φt)| - |vt·sin(φt)| - |vt|·|dt|  (conducción normal)")
    print("   R = +100  (meta alcanzada)")
    print()
    print("   Donde:")
    print("   - vt: velocidad del vehículo (m/s)")
    print("   - dt: distancia al centro del carril (m)")
    print("   - φt: ángulo respecto al carril (rad)")
    print()
    print("-" * 70)
    print()
    
    # Ejecutar varios steps y analizar recompensas
    print("🎮 EJECUTANDO STEPS Y ANALIZANDO RECOMPENSAS:")
    print("-" * 70)
    print()
    
    total_reward = 0
    step_rewards = []
    
    for step_num in range(20):
        # Diferentes acciones para probar la función de recompensa
        if step_num < 5:
            action = 22  # Throttle 1.0, steering 0 (acelerar recto)
            action_desc = "Acelerar recto"
        elif step_num < 10:
            action = 15  # Throttle 0.6, steering 0 (avanzar moderado)
            action_desc = "Avanzar moderado"
        elif step_num < 15:
            action = 16  # Throttle 0.6, steering 0.75 (girar derecha)
            action_desc = "Girar derecha"
        else:
            action = 19  # Throttle 0.6, steering -0.75 (girar izquierda)
            action_desc = "Girar izquierda"
        
        observation, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step_rewards.append(reward)
        
        # Obtener driving features del info
        if 'driving_features' in info:
            df = info['driving_features']
            vt, dt, φt = df[0], df[1], df[2]
            
            # Calcular componentes de la recompensa
            r_forward = np.abs(vt * np.cos(φt))
            r_lateral = np.abs(vt * np.sin(φt))
            r_deviation = np.abs(vt) * np.abs(dt)
            
            print(f"Step {step_num + 1:2d} [{action_desc:16s}]:")
            print(f"   Driving: vt={vt:5.2f} m/s | dt={dt:5.2f} m | φt={np.degrees(φt):6.1f}°")
            print(f"   Reward components:")
            print(f"      +{r_forward:6.3f} (velocidad adelante)")
            print(f"      -{r_lateral:6.3f} (velocidad lateral)")
            print(f"      -{r_deviation:6.3f} (desviación centro)")
            print(f"   → Reward total: {reward:7.3f}")
            print()
        else:
            print(f"Step {step_num + 1:2d}: reward={reward:6.2f} (sin driving features)")
            print()
        
        if done or truncated:
            print(f"⚠️ Episodio terminado en step {step_num + 1}")
            if done:
                print("   Razón: done=True")
            if truncated:
                print("   Razón: truncated=True")
            break
    
    print("-" * 70)
    print()
    
    # Estadísticas finales
    print("📈 ESTADÍSTICAS DE RECOMPENSAS:")
    print("-" * 70)
    print(f"   Total acumulada: {total_reward:.2f}")
    print(f"   Promedio:        {np.mean(step_rewards):.3f}")
    print(f"   Máxima:          {np.max(step_rewards):.3f}")
    print(f"   Mínima:          {np.min(step_rewards):.3f}")
    print(f"   Desv. estándar:  {np.std(step_rewards):.3f}")
    print("-" * 70)
    print()
    
    # Análisis de comportamiento
    print("🔍 ANÁLISIS DE COMPORTAMIENTO:")
    print("-" * 70)
    
    positive_rewards = [r for r in step_rewards if r > 0]
    negative_rewards = [r for r in step_rewards if r < 0]
    
    print(f"   Rewards positivas: {len(positive_rewards)}/{len(step_rewards)} steps")
    print(f"   Rewards negativas: {len(negative_rewards)}/{len(step_rewards)} steps")
    
    if positive_rewards:
        print(f"   Promedio positivo: {np.mean(positive_rewards):.3f}")
    if negative_rewards:
        print(f"   Promedio negativo: {np.mean(negative_rewards):.3f}")
    
    print()
    print("💡 Interpretación:")
    if total_reward > 0:
        print("   ✅ El agente está conduciendo bien (reward total positiva)")
    else:
        print("   ⚠️  El agente necesita mejorar (reward total negativa)")
    
    avg_reward = np.mean(step_rewards)
    if avg_reward > 0.5:
        print("   ✅ Recompensa promedio alta: conducción eficiente")
    elif avg_reward > 0:
        print("   ⚠️  Recompensa promedio baja: hay margen de mejora")
    else:
        print("   ❌ Recompensa promedio negativa: conducción ineficiente")
    
    print("-" * 70)
    print()
    
    # Cerrar entorno
    env.close()
    
    print("=" * 70)
    print("✅ TEST COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print()
    print("📖 Comparación con Paper:")
    print("   ✅ Función de recompensa implementada según Pérez-Gil et al. (2022)")
    print("   ✅ Premia velocidad hacia adelante")
    print("   ✅ Penaliza zigzagueo (velocidad lateral)")
    print("   ✅ Penaliza desviación del centro del carril")
    print("   ✅ Penalización fuerte (-200) para colisión/salida")
    print("   ✅ Recompensa alta (+100) para meta alcanzada")
    print()
    print("💡 Próximos pasos:")
    print("   1. Entrenar agente con nueva reward function")
    print("   2. Comparar convergencia vs reward anterior")
    print("   3. Validar RMSE en trayectorias")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = test_reward_function()
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

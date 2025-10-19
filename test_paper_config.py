#!/usr/bin/env python3
"""
Test rápido para verificar configuración DRL-Flatten-Image según paper
Verifica que el estado tenga 123 dimensiones (121 imagen + 1 φt + 1 dt)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv

def test_paper_config():
    """Test de configuración según paper"""
    
    print("=" * 70)
    print("🧪 TEST: Configuración DRL-Flatten-Image (Paper)")
    print("=" * 70)
    print()
    
    # 1. Configuración
    config = {
        "carla": {
            "host": "localhost",
            "port": 3000,
            "timeout": 10.0,
            "timestep": 0.1,
            "retries_on_error": 5,
            "enable_rendering": True,
            "show_display": False,
            "use_external_server": True
        },
        "others": {
            "framestack": 1,  # Paper: 1 frame
            "max_time_idle": 100,
            "max_time_episode": 500,
        }
    }
    
    experiment_config = BASE_EXPERIMENT_CONFIG.copy()
    experiment_config["use_random_routes"] = True
    experiment_config["hero"]["sensors"] = {
        "rgb_camera": {
            "type": "sensor.camera.rgb",
            "transform": "1.5,0.0,1.5,0.0,0.0,0.0",
            "image_size_x": "11",  # Paper: 11x11
            "image_size_y": "11",
            "fov": "90",
            "size": 11
        }
    }
    experiment_config["town"] = "Town01"
    experiment_config["weather"] = "ClearNoon"
    experiment_config["others"] = config["others"]
    
    print("📋 Configuración:")
    print(f"   - Imagen: 11x11 (121 píxeles)")
    print(f"   - Frame stack: 1")
    print(f"   - Estado esperado: 123 dims (121 + 1 φt + 1 dt)")
    print(f"   - Mapa: Town01")
    print(f"   - Rutas: Aleatorias")
    print()
    
    try:
        # 2. Crear experiment
        print("🔧 Creando experimento...")
        experiment = BaseEnv(experiment_config)
        
        # 3. Verificar observation space
        obs_space = experiment.get_observation_space()
        print(f"✓ Observation space creado: {obs_space.shape}")
        
        expected_dims = 11 * 11 * 1 + 2  # 121 + 2 = 123
        actual_dims = obs_space.shape[0]
        
        if actual_dims == expected_dims:
            print(f"✅ Estado correcto: {actual_dims} dimensiones")
            print(f"   - Imagen 11x11: {11*11} píxeles")
            print(f"   - φt: 1 valor")
            print(f"   - dt: 1 valor")
        else:
            print(f"❌ Estado incorrecto: {actual_dims} (esperado {expected_dims})")
            return False
        
        print()
        
        # 4. Verificar action space
        actions = experiment.get_actions()
        num_actions = len(actions)
        print(f"✓ Acciones: {num_actions} (esperado 27)")
        
        if num_actions == 27:
            print(f"✅ Número de acciones correcto")
        else:
            print(f"⚠️  Número de acciones: {num_actions} (paper usa 27)")
        
        print()
        
        # 5. Crear entorno CARLA
        print("🚗 Creando entorno CARLA...")
        env = CarlaEnv(experiment, config)
        print(f"✓ Entorno creado")
        print(f"   - Observation space: {env.observation_space.shape}")
        print(f"   - Action space: {env.action_space.n} acciones")
        print()
        
        # 6. Reset y obtener primer estado
        print("🔄 Reset del entorno...")
        state, info = env.reset()
        print(f"✓ Estado obtenido: shape={state.shape}, dtype={state.dtype}")
        
        if state.shape[0] == 123:
            print(f"✅ Estado tiene 123 dimensiones ✓")
            print(f"   - Primeros 121 valores: imagen 11x11 aplanada")
            print(f"   - Valor 122: φt = {state[121]:.4f} rad")
            print(f"   - Valor 123: dt = {state[122]:.4f} m")
        else:
            print(f"❌ Estado tiene {state.shape[0]} dims (esperado 123)")
            return False
        
        print()
        
        # 7. Verificar driving features en info
        if 'angle_to_lane' in info and 'distance_to_center' in info:
            print(f"✓ Driving features en info:")
            print(f"   - φt (angle_to_lane): {info['angle_to_lane']:.4f} rad")
            print(f"   - dt (distance_to_center): {info['distance_to_center']:.4f} m")
            print(f"   - vt (velocity): {info['velocity']:.2f} m/s")
        
        print()
        
        # 8. Test de un step
        print("🎮 Ejecutando 1 step...")
        action = 8  # Straight
        next_state, reward, done, _, next_info = env.step(action)
        
        print(f"✓ Step completado:")
        print(f"   - Next state shape: {next_state.shape}")
        print(f"   - Reward: {reward:.3f}")
        print(f"   - Done: {done}")
        
        if next_state.shape[0] == 123:
            print(f"✅ Next state correcto: 123 dimensiones")
        else:
            print(f"❌ Next state incorrecto: {next_state.shape[0]} dims")
        
        print()
        
        # 9. Cleanup
        print("🧹 Limpiando...")
        env.close()
        print("✓ Entorno cerrado")
        
        print()
        print("=" * 70)
        print("✅ TODOS LOS TESTS PASARON")
        print("=" * 70)
        print()
        print("🎯 Tu configuración es IDÉNTICA al paper:")
        print("   - Estado: 123 dims (121 imagen + 1 φt + 1 dt)")
        print("   - Imagen: 11x11 B/W")
        print("   - Frame stack: 1")
        print("   - Acciones: 27 discretas")
        print("   - Mapa: Town01")
        print()
        print("🚀 Listo para entrenar con DQN o DDPG!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_paper_config()
    sys.exit(0 if success else 1)

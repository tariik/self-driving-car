"""
Test script para verificar que las Driving Features funcionan correctamente
Como en el paper de Pérez-Gil et al. (2022)
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv

def test_driving_features():
    """Test para verificar driving features: vt, dt, φt"""
    
    print("=" * 60)
    print("🧪 TEST: DRIVING FEATURES (Paper Implementation)")
    print("=" * 60)
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
    
    # Verificar driving features en info
    print("📊 VERIFICANDO DRIVING FEATURES:")
    print("-" * 60)
    
    if 'driving_features' in info:
        df = info['driving_features']
        print(f"✅ Driving features encontradas!")
        print()
        print(f"   📐 Shape: {df.shape}")
        print(f"   📊 Dtype: {df.dtype}")
        print()
        print(f"   🚀 vt (velocidad):           {df[0]:.3f} m/s")
        print(f"   📏 dt (distancia al centro): {df[1]:.3f} m")
        print(f"   🔄 φt (ángulo al carril):    {df[2]:.3f} rad ({np.degrees(df[2]):.1f}°)")
        print()
        
        # También verificar en formato individual
        print("   💡 Info adicional:")
        print(f"      - velocity:          {info.get('velocity', 'N/A')} m/s")
        print(f"      - distance_to_center: {info.get('distance_to_center', 'N/A')} m")
        print(f"      - angle_to_lane:      {info.get('angle_to_lane', 'N/A')} rad")
        
    else:
        print("❌ ERROR: Driving features NO encontradas en info")
        print(f"   Info keys: {info.keys()}")
        return False
    
    print()
    print("-" * 60)
    
    # Ejecutar varios steps para ver cómo cambian
    print()
    print("🎮 EJECUTANDO STEPS PARA VERIFICAR CAMBIOS:")
    print("-" * 60)
    
    for step_num in range(5):
        # Acción: avanzar recto con aceleración moderada
        action = 15  # Straight con throttle 0.6
        
        observation, reward, done, truncated, info = env.step(action)
        
        if 'driving_features' in info:
            df = info['driving_features']
            vt, dt, φt = df[0], df[1], df[2]
            
            print(f"   Step {step_num + 1}: vt={vt:6.2f} m/s  |  dt={dt:5.2f} m  |  φt={np.degrees(φt):6.1f}°  |  reward={reward:6.2f}")
        else:
            print(f"   Step {step_num + 1}: ❌ No driving features")
        
        if done or truncated:
            print(f"   ⚠️ Episodio terminado en step {step_num + 1}")
            break
    
    print("-" * 60)
    print()
    
    # Cerrar entorno
    env.close()
    
    print("=" * 60)
    print("✅ TEST COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print()
    print("📖 Comparación con Paper:")
    print("   Paper usa: state = (visual_features, vt, dt, φt)")
    print("   Tu código: state = frame_stack (visual)")
    print("              info = {'driving_features': [vt, dt, φt]}")
    print()
    print("💡 Próximo paso: Usar driving_features en el agente DRL")
    print("   - Concatenar con frame_stack")
    print("   - Usar en reward function")
    print("   - Agregar a red neuronal")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = test_driving_features()
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

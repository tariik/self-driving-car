import os
import sys

import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.agents.drl_flatten_agent import DRLFlattenAgent
from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv
from src.utils.video_recorder import VideoRecorder
from src.utils.display_manager import DisplayManager


def main():
    # 1. Set up the configuration
    config = {
        "carla": {
            "host": "localhost",
            "port": 3000,  # Use the port where CARLA is already running
            "timeout": 60.0,  # Increased timeout for slow connections
            "timestep": 0.1,
            "retries_on_error": 10,  # Increased retries
            "resolution_x": 800,  # Render window width
            "resolution_y": 600,  # Render window height
            "quality_level": "Low",  # Low quality for better performance
            "enable_rendering": True,  # Enable camera rendering
            "show_display": False,  # Can't show CARLA window without X server
            "use_external_server": True  # Use the existing CARLA server
        },
        "others": {
            "framestack": 1,  # Paper: 1 frame (no stacking)
            "max_time_idle": 100,
            "max_time_episode": 500,
        }
    }
    
    # 2. Extend the base experiment config with custom settings
    experiment_config = BASE_EXPERIMENT_CONFIG.copy()
    
    # ✅ ACTIVAR RUTAS ALEATORIAS Y VISUALIZACIÓN DE WAYPOINTS
    experiment_config["use_random_routes"] = True  # Genera rutas aleatorias
    
    experiment_config["hero"]["sensors"] = {
        "rgb_camera": {
            "type": "sensor.camera.rgb",
            # CÁMARA FRONTAL apuntando hacia la CARRETERA
            # Posición: En el capó, mirando hacia adelante y ABAJO
            # x=2.0 (adelante), y=0.0 (centrada), z=1.2 (altura media)
            # pitch=-25 (25° hacia abajo para VER MÁS CARRETERA y evitar cielo/edificios)
            "transform": "2.0,0.0,1.2,0.0,-25.0,0.0",  # x,y,z,roll,pitch,yaw
            # PAPER: Captura original 640×480, luego resize a 11×11
            # "from 640x480 pixels to 11x11, reducing the amount of data from 300k to 121"
            "image_size_x": "640",  # Paper: Captura original en 640×480
            "image_size_y": "480",  # Paper: Captura original en 640×480
            "fov": "90",  # Campo de visión estándar para conducción
            "size": 11  # Target resize: 11×11 = 121 pixels (hecho en post_process_image)
        }
    }
    # Use default spawn points instead of a specific one
    experiment_config["hero"]["spawn_points"] = []  # Empty list will use map spawn points
    experiment_config["town"] = "Town01"  # Mapa del paper original (Pérez-Gil et al. 2022)
    experiment_config["weather"] = "ClearNoon"
    experiment_config["others"] = config["others"]
    
    # 3. Create the experiment and environment
    print("Creating experiment...")
    experiment = BaseEnv(experiment_config)
    print("Creating environment...")
    env = CarlaEnv(experiment, config)
    print("Environment created successfully!")
    
    #env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    
    print("Creating agent...")
    agent = DRLFlattenAgent(env, seed=0)
    print("Agent created successfully!")

    print("Resetting environment...")
    state, _ = env.reset()
    # Removido segundo reset duplicado que causaba spawn en mala ubicación
    print(f"State type: {type(state)}")
    print(f"State shape: {state.shape}")
    print(f"State size: {np.prod(state.shape)}")
    # Before agent initialization
    flat_size = int(np.prod(env.observation_space.shape))
    print(f"Flattened observation space size: {flat_size}")
    
    # Initialize display with CARLA camera (like manual_control.py)
    print("Initializing display with CARLA camera...")
    use_display = True  # Set to False to use only video recorder
    
    display = None
    if use_display:
        try:
            display = DisplayManager(
                world=env.core.world,
                hero_vehicle=env.hero,
                width=800,
                height=600,
                follow_spectator=True  # Mostrar la MISMA vista que el spectator
            )
        except Exception as e:
            print(f"   ⚠️  Could not open display: {e}")
            print(f"   📹 Falling back to video recorder only")
            use_display = False
    
    # Always use video recorder as backup
    print("Initializing video recorder...")
    recorder = VideoRecorder(output_dir='render_output', fps=10)
    
    print("Running simulation...")
    total_reward = 0.0
    try:
        for j in range(5000):
            # Check if user wants to quit (only if display is active)
            if display and display.process_events():
                print("User requested quit")
                break
            
            action = agent.act(state)
            env.render()  # ✅ ACTIVADO: Guardar imagen 11×11 que ve el agente
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # Update display if available
            if display:
                display.update(
                    step=j+1,
                    reward=reward,
                    total_reward=total_reward,
                    done=done
                )
            
            # Record frame with HUD info
            recorder.add_frame(
                observation=state,
                step=j+1,
                reward=reward,
                total_reward=total_reward,
                done=done
            )
            
            print(f"Step {j+1}: reward={reward}, done={done}")
            if done:
                break
    finally:
        print("Closing environment...")
        if display:
            display.close()
        recorder.close()
        env.close()
    print("Done!")

    

if __name__ == "__main__":
    main()

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
    
    # ==========================================
    # TRAINING LOOP - Según paper Pérez-Gil et al. 2022
    # ==========================================
    print("\n" + "="*80)
    print("🚀 INICIANDO ENTRENAMIENTO DRL-Flatten-Image")
    print("="*80)
    
    # Hyperparameters según el paper
    MAX_EPISODES = 500  # Paper: DQN necesita ~20K, DDPG solo ~500
    MAX_STEPS_PER_EPISODE = 500
    SAVE_EVERY_N_EPISODES = 50  # Guardar checkpoint cada 50 episodios
    
    # Desactivar guardado de renders (consume mucho espacio y tiempo)
    SAVE_RENDERS = False
    DEBUG_STATE_FIRST_3_STEPS = False  # Desactivar debug verbose
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    best_episode = 0
    
    try:
        for episode in range(MAX_EPISODES):
            # Reset environment for new episode with random route
            state, _ = env.reset()
            episode_reward = 0.0
            episode_step = 0
            
            print(f"\n📍 Episode {episode+1}/{MAX_EPISODES}")
            
            for step in range(MAX_STEPS_PER_EPISODE):
                # Check if user wants to quit (only if display is active)
                if display and display.process_events():
                    print("User requested quit")
                    break
                
                # 🔍 DEBUG: Solo para los primeros 3 steps del primer episodio
                if DEBUG_STATE_FIRST_3_STEPS and episode == 0 and step < 3:
                    print(f"\n{'='*80}")
                    print(f"🎯 ESTADO ANTES DE AGENTE (Episode {episode+1}, Step {step+1})")
                    print(f"{'='*80}")
                    # print(f"📊 Estado shape: {state.shape}")
                    image_flat = state[:121]
                    phi_t = state[121] if len(state) > 121 else 0.0
                    d_t = state[122] if len(state) > 122 else 0.0
                    image_11x11 = image_flat.reshape(11, 11)
                    image_display = ((image_11x11 * 128) + 128).clip(0, 255).astype(np.uint8)
                    h_var = np.mean([np.std(row) for row in image_display])
                    print(f"   H-Var: {h_var:.1f}, φt: {np.degrees(phi_t):.2f}°, dt: {d_t:.3f}m")
                    print(f"{'='*80}\n")
                
                # Agent selects action
                action = agent.act(state)
                
                # Guardar render solo si está activado (desactivado por defecto)
                if SAVE_RENDERS:
                    env.render()
                
                # Execute action
                next_state, reward, done, _, _ = env.step(action)
                
                # Agent learns from experience
                agent.step(state, action, reward, next_state, done)
                
                # Update state and statistics
                state = next_state
                episode_reward += reward
                episode_step += 1
                
                # Update display if available
                if display:
                    display.update(
                        step=episode_step,
                        reward=reward,
                        total_reward=episode_reward,
                        done=done
                    )
                
                # Record frame with video recorder
                recorder.add_frame(
                    observation=state,
                    step=episode_step,
                    reward=reward,
                    total_reward=episode_reward,
                    done=done
                )
                
                # Episode termination
                if done:
                    print(f"   ✓ Episode finished at step {episode_step}: Total reward = {episode_reward:.2f}")
                    break
            
            # End of episode - save statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_step)
            
            # Check if this is the best episode so far
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_episode = episode + 1
                print(f"   🏆 NEW BEST! Episode {episode+1}: Reward = {episode_reward:.2f}")
            
            # Save checkpoint every N episodes
            if (episode + 1) % SAVE_EVERY_N_EPISODES == 0:
                checkpoint_path = f'checkpoints/drl_flatten_episode_{episode+1}.pth'
                agent.save(checkpoint_path)
                print(f"   💾 Checkpoint saved: {checkpoint_path}")
                print(f"   📊 Avg reward (last {SAVE_EVERY_N_EPISODES}): {np.mean(episode_rewards[-SAVE_EVERY_N_EPISODES:]):.2f}")
            
            # Print episode summary every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"\n📈 Episodes {episode-8}-{episode+1} Summary:")
                print(f"   Avg Reward: {avg_reward:.2f}")
                print(f"   Avg Length: {avg_length:.1f} steps")
                print(f"   Best so far: Episode {best_episode} with {best_reward:.2f}")
        
        # Training complete
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED!")
        print("="*80)
        print(f"Total episodes: {MAX_EPISODES}")
        print(f"Best episode: {best_episode} with reward {best_reward:.2f}")
        print(f"Average reward (all episodes): {np.mean(episode_rewards):.2f}")
        
        # Save final model
        final_model_path = 'checkpoints/drl_flatten_final.pth'
        agent.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        print("Closing environment...")
        if display:
            display.close()
        recorder.close()
        env.close()
    print("Done!")

    

if __name__ == "__main__":
    main()

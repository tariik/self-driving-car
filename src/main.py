import os
import sys
import time

import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.agents.drl_flatten_agent import DRLFlattenAgent
from src.env.base_env import BASE_EXPERIMENT_CONFIG, BaseEnv
from src.env.carla_env import CarlaEnv
from src.utils.video_recorder import VideoRecorder
from src.utils.display_manager import DisplayManager
from src.utils.tensorboard_logger import TensorBoardLogger


def extract_state_info(state, env):
    """
    Extrae información del estado y del vehículo de manera eficiente.
    Optimización: Calcula todo de una vez en lugar de accesos múltiples.
    
    Returns:
        tuple: (v_t, d_t, phi_t, action_control)
    """
    # Extraer características del estado
    phi_t = state[121] if len(state) > 121 else 0.0  # Ángulo (rad)
    d_t = state[122] if len(state) > 122 else 0.0    # Distancia (m)
    
    # Obtener velocidad del vehículo (optimizado)
    if hasattr(env, 'hero'):
        vel = env.hero.get_velocity()
        v_t = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # Más rápido que np.linalg.norm
    else:
        v_t = 0.0
    
    return v_t, d_t, phi_t


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
    experiment_config["clean_road"] = True  # Eliminar todos los vehículos del mapa al iniciar
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
    # ⚡ OPTIMIZACIÓN: Desactivar display para entrenamiento rápido
    # Cambiar a True solo para debugging/visualización
    use_display = False  # TRAINING: Desactivado para máximo rendimiento (cambiar a True para ver)
    
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
    
    # ⚡ OPTIMIZACIÓN: Monitorear rendimiento
    training_start_time = time.time()
    
    # Hyperparameters según el paper
    MAX_EPISODES = 20000  # Paper: DDPG necesita ~500 episodios (DQN necesita 20,000)
    MAX_STEPS_PER_EPISODE = 3000  # Paper: rutas 180-700m, ~10 FPS, ~5-15 m/s → ~1000-2000 steps típicos
    SAVE_EVERY_N_EPISODES = 50  # Guardar checkpoint cada 50 episodios
    
    # 🎯 EPSILON-GREEDY PARAMETERS (para DQN exploration)
    eps_start = 1.0      # Epsilon inicial (100% exploración al inicio)
    eps_end = 0.01       # Epsilon final (1% exploración al final)
    eps_decay = 0.995    # Decaimiento exponencial por episodio
    eps = eps_start      # Epsilon actual
    
    # Controles de guardado para training vs evaluation
    SAVE_RENDERS = False  # Guardar frames individuales (desactivado para training)
    SAVE_VIDEO = False  # Crear video al final (desactivado para training)
    DEBUG_STATE_FIRST_3_STEPS = False  # Desactivar debug verbose
    
    # ⚡ OPTIMIZACIÓN: Frecuencia de logs
    LOG_FREQUENCY = 100  # Mostrar logs cada 100 steps (en lugar de cada 10)
    
    # 📊 TensorBoard Logger
    experiment_name = f"DQN_Flatten_{time.strftime('%Y%m%d_%H%M%S')}"
    tb_logger = TensorBoardLogger(log_dir='runs', experiment_name=experiment_name)
    
    # Log hiperparámetros del experimento
    tb_logger.log_hyperparameters({
        'algorithm': 'DQN',
        'agent': 'DRL-Flatten-Image',
        'max_episodes': MAX_EPISODES,
        'max_steps_per_episode': MAX_STEPS_PER_EPISODE,
        'buffer_size': 100000,
        'batch_size': 32,
        'gamma': 0.99,
        'learning_rate': 0.0005,
        'tau': 0.001,
        'update_every': 4,
        'eps_start': eps_start,
        'eps_end': eps_end,
        'eps_decay': eps_decay,
        'state_size': 123,
        'action_size': 27,
        'image_resolution': '640x480',
        'image_resize': '11x11',
    })
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    best_episode = 0
    
    # Contadores para TensorBoard
    global_step = 0
    collision_episodes = []
    lane_invasion_episodes = []
    
    try:
        for episode in range(MAX_EPISODES):
            # ⚡ Timer para este episodio
            episode_start_time = time.time()
            
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
                
                # 🎯 Agent selects action (con epsilon-greedy para exploración)
                action = agent.act(state, eps)
                
                # ⚡ OPTIMIZACIÓN: Extraer información del estado eficientemente
                v_t, d_t, phi_t = extract_state_info(state, env)
                
                # Obtener control de acción (throttle, steer, brake)
                action_control = env.experiment.get_actions()[int(action)]
                throttle = action_control[0]
                steer = action_control[1]
                brake = action_control[2]
                
                # Guardar render solo si está activado (desactivado por defecto)
                if SAVE_RENDERS:
                    env.render()
                
                # Execute action
                next_state, reward, done, _, _ = env.step(action)
                
                # 📊 LOG PROFESIONAL EN CONSOLA (frecuencia configurable)
                if step % LOG_FREQUENCY == 0:
                    print(f"\n{'='*70}")
                    print(f"  AGENT STATUS - Step {episode_step+1:4d}")
                    print(f"{'='*70}")
                    print(f"  Vehicle State:")
                    print(f"    • Velocity:    {v_t:6.2f} m/s")
                    print(f"    • Distance:    {d_t:6.3f} m     (to lane center)")
                    print(f"    • Angle:       {np.degrees(phi_t):+6.2f}°    (lane alignment)")
                    print(f"  ───────────────────────────────────────────────────────────────")
                    print(f"  Control Action:")
                    print(f"    • Throttle:    {throttle:5.3f}      (acceleration)")
                    print(f"    • Steering:    {steer:+6.3f}      (direction)")
                    print(f"    • Brake:       {brake:5.3f}      (braking)")
                    print(f"  ───────────────────────────────────────────────────────────────")
                    print(f"  Training Metrics:")
                    print(f"    • Reward:      {reward:+8.4f}")
                    print(f"    • Total:       {episode_reward+reward:+8.2f}")
                    print(f"{'='*70}\n")
                
                # Agent learns from experience
                agent.step(state, action, reward, next_state, done)
                
                # 📊 TensorBoard: Log step metrics (cada 10 steps para reducir overhead)
                if global_step % 10 == 0:
                    tb_logger.log_step(
                        step=global_step,
                        reward=reward,
                        velocity=v_t,
                        distance=d_t,
                        angle=phi_t,
                        throttle=throttle,
                        steer=steer,
                        brake=brake
                    )
                
                # Update state and statistics
                state = next_state
                episode_reward += reward
                episode_step += 1
                global_step += 1
                
                # Update display if available
                if display:
                    display.update(
                        step=episode_step,
                        reward=reward,
                        total_reward=episode_reward,
                        done=done,
                        velocity=v_t,
                        distance=d_t,
                        angle=phi_t,
                        throttle=throttle,
                        steer=steer,
                        brake=brake,
                        epsilon=eps  # 🎯 Mostrar tasa de exploración DQN
                    )
                
                # Record frame with video recorder (solo si SAVE_VIDEO está activado)
                if SAVE_VIDEO:
                    recorder.add_frame(
                        observation=state,
                        step=episode_step,
                        reward=reward,
                        total_reward=episode_reward,
                        done=done
                    )
                
                # Episode termination
                if done:
                    episode_time = time.time() - episode_start_time
                    print(f"   ✓ Episode finished at step {episode_step}: Total reward = {episode_reward:.2f}")
                    print(f"   ⏱️  Episode time: {episode_time:.1f}s ({episode_step/episode_time:.1f} steps/s)")
                    break
            
            # End of episode - save statistics
            episode_time = time.time() - episode_start_time
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_step)
            
            # Detectar colisión e invasión de carril (reward -200 indica terminación por error)
            had_collision = 1 if (done and episode_reward < -100) else 0
            had_lane_invasion = 0  # Podría detectarse con sensores adicionales
            
            # 📊 TensorBoard: Log episode metrics
            tb_logger.log_episode(
                episode=episode + 1,
                total_reward=episode_reward,
                episode_length=episode_step,
                collisions=had_collision,
                lane_invasions=had_lane_invasion,
                epsilon=eps
            )
            
            # Log running averages (ventana de 10 y 100 episodios)
            if (episode + 1) >= 10:
                tb_logger.log_running_avg(episode + 1, episode_rewards, episode_lengths, window_size=10)
            if (episode + 1) >= 100:
                tb_logger.log_running_avg(episode + 1, episode_rewards, episode_lengths, window_size=100)
            
            # 🎯 Epsilon decay (reducir exploración gradualmente)
            eps = max(eps_end, eps_decay * eps)
            
            # Check if this is the best episode so far
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_episode = episode + 1
                print(f"   🏆 NEW BEST! Episode {episode+1}: Reward = {episode_reward:.2f}")
                
                # 📊 TensorBoard: Log best model
                checkpoint_path_best = f'checkpoints/drl_flatten_best.pth'
                tb_logger.log_best_model(episode + 1, episode_reward, checkpoint_path_best)
            
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
                print(f"   🎯 Epsilon: {eps:.4f} (exploration rate)")
                print(f"   Avg Reward: {avg_reward:.2f}")
                print(f"   Avg Length: {avg_length:.1f} steps")
                print(f"   Best so far: Episode {best_episode} with {best_reward:.2f}")
        
        # Training complete
        training_time = time.time() - training_start_time
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED!")
        print("="*80)
        print(f"Total episodes: {MAX_EPISODES}")
        print(f"Best episode: {best_episode} with reward {best_reward:.2f}")
        print(f"Average reward (all episodes): {np.mean(episode_rewards):.2f}")
        print(f"\n⏱️  PERFORMANCE METRICS:")
        print(f"   Total training time: {training_time/60:.1f} minutes ({training_time/3600:.2f} hours)")
        print(f"   Average per episode: {training_time/MAX_EPISODES:.1f}s")
        print(f"   Total steps: {sum(episode_lengths)}")
        print(f"   Average steps/s: {sum(episode_lengths)/training_time:.1f}")
        
        # Save final model
        final_model_path = 'checkpoints/drl_flatten_final.pth'
        agent.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # 📊 TensorBoard: Log final metrics
        tb_logger.log_hyperparameters(
            hparams_dict={
                'algorithm': 'DQN',
                'max_episodes': MAX_EPISODES,
                'eps_decay': eps_decay,
            },
            metrics_dict={
                'final/best_reward': best_reward,
                'final/avg_reward': np.mean(episode_rewards),
                'final/avg_length': np.mean(episode_lengths),
                'final/training_time_hours': training_time / 3600,
            }
        )
        
        # Create video only if enabled
        if SAVE_VIDEO:
            print("🎬 Creating training video...")
            recorder.save_video('training_video.mp4')
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        print("Closing environment...")
        
        # 📊 Close TensorBoard logger
        if 'tb_logger' in locals():
            tb_logger.close()
        
        if display:
            display.close()
        if SAVE_VIDEO:
            recorder.close()
        env.close()
    print("Done!")

    

if __name__ == "__main__":
    main()

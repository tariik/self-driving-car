"""
TensorBoard Logger for DRL-Flatten-Image Training
Based on Pérez-Gil et al. (2022) metrics
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime


class TensorBoardLogger:
    """
    TensorBoard logger para seguimiento de métricas clave de entrenamiento.
    
    Métricas principales según el paper:
    1. Reward acumulado por episodio
    2. Longitud de episodio (steps)
    3. Velocidad del vehículo (vt)
    4. Distancia al centro del carril (dt)
    5. Ángulo con respecto al carril (φt)
    6. Tasa de colisiones
    7. Tasa de invasión de carril
    8. Epsilon (exploración para DQN)
    9. Loss de la red (si disponible)
    """
    
    def __init__(self, log_dir='runs', experiment_name=None):
        """
        Inicializa el logger de TensorBoard.
        
        Args:
            log_dir: Directorio base para logs
            experiment_name: Nombre del experimento (si None, usa timestamp)
        """
        if experiment_name is None:
            experiment_name = f"DRL_Flatten_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_path = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_path)
        
        # Contadores
        self.episode_count = 0
        self.step_count = 0
        
        # Estadísticas acumuladas
        self.collision_count = 0
        self.lane_invasion_count = 0
        
        print(f"📊 TensorBoard logger initialized: {self.log_path}")
        print(f"   Run: tensorboard --logdir={log_dir}")
    
    def log_episode(self, episode, total_reward, episode_length, 
                   collisions=0, lane_invasions=0, epsilon=None):
        """
        Log métricas al final de cada episodio.
        
        Args:
            episode: Número de episodio
            total_reward: Reward acumulado en el episodio
            episode_length: Número de steps en el episodio
            collisions: Si hubo colisión (1) o no (0)
            lane_invasions: Si hubo invasión de carril (1) o no (0)
            epsilon: Valor actual de epsilon (para DQN)
        """
        # Métricas principales del episodio
        self.writer.add_scalar('Episode/Total_Reward', total_reward, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        
        # Reward promedio por step
        avg_reward_per_step = total_reward / max(episode_length, 1)
        self.writer.add_scalar('Episode/Avg_Reward_Per_Step', avg_reward_per_step, episode)
        
        # Estadísticas de colisiones e invasiones
        self.collision_count += collisions
        self.lane_invasion_count += lane_invasions
        
        self.writer.add_scalar('Episode/Collision', collisions, episode)
        self.writer.add_scalar('Episode/Lane_Invasion', lane_invasions, episode)
        
        # Tasas acumuladas
        collision_rate = self.collision_count / (episode + 1)
        lane_invasion_rate = self.lane_invasion_count / (episode + 1)
        
        self.writer.add_scalar('Cumulative/Collision_Rate', collision_rate, episode)
        self.writer.add_scalar('Cumulative/Lane_Invasion_Rate', lane_invasion_rate, episode)
        
        # Epsilon (exploración DQN)
        if epsilon is not None:
            self.writer.add_scalar('Training/Epsilon', epsilon, episode)
        
        self.episode_count = episode
    
    def log_step(self, step, reward, velocity, distance, angle, 
                throttle, steer, brake, loss=None):
        """
        Log métricas durante el episodio (cada step).
        
        Args:
            step: Step global de entrenamiento
            reward: Reward en este step
            velocity: Velocidad del vehículo vt (m/s)
            distance: Distancia al centro del carril dt (m)
            angle: Ángulo con respecto al carril φt (rad)
            throttle: Acción throttle
            steer: Acción steering
            brake: Acción brake
            loss: Loss de la red (si disponible)
        """
        # Reward instantáneo
        self.writer.add_scalar('Step/Reward', reward, step)
        
        # Estado del vehículo (driving features según paper)
        self.writer.add_scalar('State/Velocity_vt', velocity, step)
        self.writer.add_scalar('State/Distance_dt', distance, step)
        self.writer.add_scalar('State/Angle_phi_t', np.degrees(angle), step)  # En grados
        
        # Acciones de control
        self.writer.add_scalar('Action/Throttle', throttle, step)
        self.writer.add_scalar('Action/Steering', steer, step)
        self.writer.add_scalar('Action/Brake', brake, step)
        
        # Loss de la red (si disponible)
        if loss is not None:
            self.writer.add_scalar('Training/Loss', loss, step)
        
        self.step_count = step
    
    def log_running_avg(self, episode, rewards_window, lengths_window, window_size=10):
        """
        Log promedios móviles de métricas.
        
        Args:
            episode: Número de episodio
            rewards_window: Lista de rewards recientes
            lengths_window: Lista de lengths recientes
            window_size: Tamaño de la ventana para promedio
        """
        if len(rewards_window) >= window_size:
            avg_reward = np.mean(rewards_window[-window_size:])
            avg_length = np.mean(lengths_window[-window_size:])
            
            self.writer.add_scalar(f'Running_Avg_{window_size}/Reward', avg_reward, episode)
            self.writer.add_scalar(f'Running_Avg_{window_size}/Length', avg_length, episode)
    
    def log_histogram(self, tag, values, episode):
        """
        Log histograma de valores.
        
        Args:
            tag: Nombre del histograma
            values: Array de valores
            episode: Número de episodio
        """
        self.writer.add_histogram(tag, np.array(values), episode)
    
    def log_text(self, tag, text, episode):
        """
        Log texto (para notas o eventos especiales).
        
        Args:
            tag: Etiqueta del texto
            text: Contenido del texto
            episode: Número de episodio
        """
        self.writer.add_text(tag, text, episode)
    
    def log_best_model(self, episode, reward, path):
        """
        Log cuando se guarda el mejor modelo.
        
        Args:
            episode: Número de episodio
            reward: Mejor reward alcanzado
            path: Ruta donde se guardó el modelo
        """
        self.writer.add_scalar('Best/Episode', episode, episode)
        self.writer.add_scalar('Best/Reward', reward, episode)
        self.log_text('Best_Model', f'New best at episode {episode}: {reward:.2f} (saved to {path})', episode)
    
    def log_hyperparameters(self, hparams_dict, metrics_dict=None):
        """
        Log hiperparámetros del experimento.
        
        Args:
            hparams_dict: Diccionario con hiperparámetros
            metrics_dict: Diccionario con métricas finales (opcional)
        """
        if metrics_dict is None:
            metrics_dict = {}
        
        self.writer.add_hparams(hparams_dict, metrics_dict)
    
    def close(self):
        """Cierra el writer de TensorBoard."""
        self.writer.close()
        print(f"📊 TensorBoard logger closed. Logs saved to: {self.log_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

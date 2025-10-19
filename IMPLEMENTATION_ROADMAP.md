# 🔄 Comparación: Tu Proyecto vs Paper de Referencia

## 📊 TABLA COMPARATIVA

| Aspecto | Paper (Pérez-Gil et al.) | Tu Proyecto Actual | Recomendación |
|---------|--------------------------|---------------------|---------------|
| **GPU** | RTX 2080 Ti (11GB) | RTX 3080 (10GB) | ✅ Equivalente |
| **Algoritmo** | DQN + **DDPG ⭐** | DQN | ⚠️ Migrar a DDPG |
| **Cámara Posición** | Frontal (varias configs) | (1.5, 0, 1.5) frontal | ✅ CORRECTO |
| **Cámara Resolución** | 640x480 RGB | 84x84 grayscale | ⚠️ Suficiente pero diferente |
| **Estado (State)** | (waypoints/img, vt, dt, φt) | Frame stack (84×84×4) | ⚠️ Agregar driving features |
| **Acciones DQN** | 27 discretas | 29 discretas | ✅ Similar |
| **Acciones DDPG** | Continuas [-1,1], [0,1] | N/A | 🎯 Implementar |
| **Recompensa** | Compleja (3 componentes) | Básica | ⚠️ Mejorar |
| **Waypoints** | De planificador A* CARLA | No usados | 🎯 Integrar |
| **Sensores** | RGB, Lane, Collision | RGB | ⚠️ Agregar sensores |
| **Episodios (DQN)** | 8,300 - 108,600 | En progreso | ℹ️ Esperado |
| **Episodios (DDPG)** | 50 - 45,950 | N/A | 🎯 Más rápido |
| **RMSE Mejor** | 0.10m (DDPG-Waypoints) | N/A | 🎯 Meta |
| **Rutas Training** | Aleatorias cada episodio | ¿Fijas? | ⚠️ Aleatorizar |
| **Validación** | 20 iteraciones vs Ground Truth | N/A | 🎯 Implementar |

---

## 🎯 PLAN DE MEJORA PASO A PASO

### FASE 1: Mejoras Inmediatas (Sin cambiar arquitectura)

#### 1.1 Agregar Driving Features al Estado ⭐ PRIORIDAD ALTA
```python
# Estado actual
state = frame_stack  # (84, 84, 4)

# Estado mejorado (como paper)
state = {
    'visual': frame_stack,  # (84, 84, 4)
    'driving': [vt, dt, φt]  # 3 valores
}

# Obtener de CARLA
vt = vehicle.get_velocity()  # Velocidad
dt = distance_to_lane_center()  # Distancia al centro
φt = angle_to_lane_direction()  # Ángulo con carril
```

**Implementación en base_env.py:**
```python
def get_driving_features(self):
    """Extraer características de conducción como en el paper"""
    # Velocidad del vehículo
    velocity = self.hero.get_velocity()
    vt = np.sqrt(velocity.x**2 + velocity.y**2)  # m/s
    
    # Waypoint más cercano
    waypoint = self.world_map.get_waypoint(
        self.hero.get_location(),
        project_to_road=True
    )
    
    # Distancia al centro del carril
    vehicle_location = self.hero.get_location()
    lane_center = waypoint.transform.location
    dt = np.sqrt(
        (vehicle_location.x - lane_center.x)**2 + 
        (vehicle_location.y - lane_center.y)**2
    )
    
    # Ángulo respecto al carril
    vehicle_yaw = self.hero.get_transform().rotation.yaw
    lane_yaw = waypoint.transform.rotation.yaw
    φt = np.radians(vehicle_yaw - lane_yaw)
    
    return np.array([vt, dt, φt], dtype=np.float32)
```

#### 1.2 Mejorar Función de Recompensa ⭐ PRIORIDAD ALTA
```python
def calculate_reward(self, vt, dt, φt):
    """Función de recompensa del paper"""
    
    # Colisión o salida de carril
    if self.collision_sensor.triggered:
        return -200.0
    
    if self.lane_invasor_sensor.triggered:
        return -200.0
    
    # Meta alcanzada
    if self.reached_goal():
        return 100.0
    
    # Recompensa continua (ecuación del paper)
    reward = (
        np.abs(vt * np.cos(φt))      # Premia velocidad adelante
        - np.abs(vt * np.sin(φt))     # Penaliza velocidad lateral
        - np.abs(vt) * np.abs(dt)     # Penaliza desviación del centro
    )
    
    return reward
```

#### 1.3 Agregar Sensores Faltantes ⭐ PRIORIDAD MEDIA
```python
# En carla_env.py o base_env.py

def setup_sensors(self):
    """Agregar sensores como en el paper"""
    
    # 1. Sensor de colisión (ya existe implícito, hacerlo explícito)
    collision_bp = self.world.get_blueprint_library().find(
        'sensor.other.collision'
    )
    self.collision_sensor = self.world.spawn_actor(
        collision_bp,
        carla.Transform(),
        attach_to=self.hero
    )
    self.collision_sensor.listen(
        lambda event: self._on_collision(event)
    )
    
    # 2. Sensor de invasión de carril
    lane_invasion_bp = self.world.get_blueprint_library().find(
        'sensor.other.lane_invasion'
    )
    self.lane_invasor = self.world.spawn_actor(
        lane_invasion_bp,
        carla.Transform(),
        attach_to=self.hero
    )
    self.lane_invasor.listen(
        lambda event: self._on_lane_invasion(event)
    )
    
def _on_collision(self, event):
    self.collision_triggered = True
    
def _on_lane_invasion(self, event):
    # Solo penalizar salidas significativas
    for marking in event.crossed_lane_markings:
        if marking.type != carla.LaneMarkingType.NONE:
            self.lane_invasion_triggered = True
```

#### 1.4 Rutas Aleatorias en Entrenamiento ⭐ PRIORIDAD MEDIA
```python
def reset(self):
    """Reset con ruta aleatoria como en el paper"""
    
    # Obtener dos puntos aleatorios del mapa
    spawn_points = self.world_map.get_spawn_points()
    start_idx = np.random.randint(0, len(spawn_points))
    end_idx = np.random.randint(0, len(spawn_points))
    
    while end_idx == start_idx:
        end_idx = np.random.randint(0, len(spawn_points))
    
    # Usar A* planner de CARLA
    self.route_waypoints = self._get_route(
        spawn_points[start_idx].location,
        spawn_points[end_idx].location
    )
    
    # Spawn vehículo en punto inicial
    self.hero.set_transform(spawn_points[start_idx])
    
    return self._get_observation()

def _get_route(self, start, end):
    """Obtener ruta con A* como en el paper"""
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    
    planner = GlobalRoutePlanner(self.world_map, 2.0)
    route = planner.trace_route(start, end)
    waypoints = [waypoint for waypoint, _ in route]
    
    return waypoints
```

---

### FASE 2: Implementar Agente DDPG ⭐⭐⭐ PRIORIDAD MÁXIMA

#### 2.1 ¿Por qué DDPG?

**Ventajas demostradas en el paper:**
- ✅ **50x más rápido**: Converge en ~150 episodios vs 8,300+ de DQN
- ✅ **Mejor RMSE**: 0.10m vs 0.21m (DQN)
- ✅ **Control continuo**: Steering/throttle suave (más realista)
- ✅ **Menos recursos**: Menos episodios = menos tiempo GPU
- ✅ **Transferencia**: Más fácil mover a vehículo real

**Resultados del paper (ruta 180m):**
```
DQN-Waypoints:  RMSE=0.21m, Tiempo=29.3s, Episodios=8,300
DDPG-Waypoints: RMSE=0.13m, Tiempo=20.6s, Episodios=150  ⭐ 
```

#### 2.2 Arquitectura DDPG a Implementar

```python
# src/agents/ddpg_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    """Red Actor: State → Action (continua)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        # Para Frame Stack + Driving Features
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calcular dimensión después de CNN
        cnn_out_dim = 64 * 7 * 7  # Para 84x84 input
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 3, hidden_dim),  # +3 para (vt, dt, φt)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Salida [-1, 1]
        )
        
    def forward(self, frames, driving_features):
        # frames: (batch, 4, 84, 84)
        # driving_features: (batch, 3)
        
        cnn_out = self.cnn(frames)
        combined = torch.cat([cnn_out, driving_features], dim=1)
        actions = self.fc(combined)
        
        return actions

class Critic(nn.Module):
    """Red Critic: (State, Action) → Q-Value"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_dim = 64 * 7 * 7
        
        # Combina estado + acción
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 3 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Q-value escalar
        )
        
    def forward(self, frames, driving_features, actions):
        cnn_out = self.cnn(frames)
        combined = torch.cat([cnn_out, driving_features, actions], dim=1)
        q_value = self.fc(combined)
        
        return q_value

class DDPGAgent:
    """Agente DDPG como en el paper"""
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.001,
        buffer_size=1000000,
        batch_size=64
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Redes principales
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        
        # Redes target (como en el paper)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # Copiar pesos iniciales
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # Hiperparámetros
        self.gamma = gamma
        self.tau = tau
        
    def select_action(self, state, noise=0.1):
        """Seleccionar acción con exploración (Ornstein-Uhlenbeck)"""
        frames, driving_features = self._parse_state(state)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(frames, driving_features).cpu().numpy()[0]
        self.actor.train()
        
        # Agregar ruido para exploración
        action += noise * np.random.randn(len(action))
        action = np.clip(action, -1, 1)
        
        return action
    
    def train(self):
        """Entrenar actor y critic"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertir a tensors
        # ... (implementación completa)
        
        # Actualizar Critic
        # ... (como en el paper: TD error)
        
        # Actualizar Actor
        # ... (como en el paper: policy gradient)
        
        # Soft update de targets
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
    
    def _soft_update(self, local_model, target_model):
        """Soft update de target networks"""
        for target_param, local_param in zip(
            target_model.parameters(), 
            local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + 
                (1.0 - self.tau) * target_param.data
            )
```

#### 2.3 Acciones Continuas (DDPG)
```python
def apply_ddpg_action(self, action):
    """
    Aplicar acción continua de DDPG
    action: numpy array [-1, 1] x 2 (steering, throttle)
    """
    steering = float(action[0])  # [-1, 1]
    throttle = float((action[1] + 1) / 2)  # [-1,1] → [0,1]
    
    control = carla.VehicleControl(
        throttle=throttle,
        steer=steering,
        brake=0.0,  # Como en el paper
        hand_brake=False,
        manual_gear_shift=False
    )
    
    self.hero.apply_control(control)
```

---

### FASE 3: Implementar Agente con Waypoints ⭐⭐ MEJOR DEL PAPER

#### 3.1 DRL-Carla-Waypoints (DDPG)

Este fue el **mejor agente** en el paper: RMSE = 0.10m

```python
# src/agents/ddpg_waypoints_agent.py

class WaypointsActor(nn.Module):
    """
    Actor para waypoints (más simple que CNN)
    Como en el paper: 15 waypoints + driving features
    """
    def __init__(self, action_dim=2, hidden_dim=256):
        super(WaypointsActor, self).__init__()
        
        # Input: 15 waypoints (solo coord X) + 3 driving features
        input_dim = 15 + 3  # 18 total
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
    def forward(self, waypoints, driving_features):
        # waypoints: (batch, 15) - solo coord X local
        # driving_features: (batch, 3)
        
        combined = torch.cat([waypoints, driving_features], dim=1)
        actions = self.fc(combined)
        
        return actions

def get_local_waypoints(self, num_waypoints=15):
    """
    Obtener waypoints en coordenadas locales del vehículo
    Como en el paper: transformación a referencia local
    """
    # Posición y orientación actual
    vehicle_transform = self.hero.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    
    # Waypoint actual
    current_waypoint = self.world_map.get_waypoint(
        vehicle_location,
        project_to_road=True
    )
    
    # Obtener próximos waypoints
    waypoints_global = []
    waypoint = current_waypoint
    
    for _ in range(num_waypoints):
        waypoint_list = waypoint.next(2.0)  # 2m adelante
        if not waypoint_list:
            break
        waypoint = waypoint_list[0]
        waypoints_global.append(waypoint)
    
    # Transformar a coordenadas locales (como en el paper)
    waypoints_local = []
    
    yaw = np.radians(vehicle_rotation.yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    for wp in waypoints_global:
        # Trasladar
        dx = wp.transform.location.x - vehicle_location.x
        dy = wp.transform.location.y - vehicle_location.y
        
        # Rotar (aplicar matriz de transformación del paper)
        x_local = cos_yaw * dx + sin_yaw * dy
        y_local = -sin_yaw * dx + cos_yaw * dy
        
        # Solo usar coordenada X (lateral) como en el paper
        waypoints_local.append(x_local)
    
    # Pad si faltan waypoints
    while len(waypoints_local) < num_waypoints:
        waypoints_local.append(waypoints_local[-1] if waypoints_local else 0.0)
    
    return np.array(waypoints_local[:num_waypoints], dtype=np.float32)
```

---

### FASE 4: Validación como en el Paper

#### 4.1 Métrica RMSE
```python
def validate_agent(agent, route, num_iterations=20):
    """
    Validar agente como en el paper:
    - Misma ruta 20 veces
    - Comparar con ground truth
    - Calcular RMSE
    """
    errors = []
    times = []
    max_error = 0.0
    
    for i in range(num_iterations):
        print(f"Iteración {i+1}/{num_iterations}")
        
        # Reset en la ruta específica
        env.reset_with_route(route)
        
        trajectory = []
        start_time = time.time()
        done = False
        
        while not done:
            state = env.get_state()
            action = agent.select_action(state, noise=0)  # Sin ruido
            next_state, reward, done, info = env.step(action)
            
            # Guardar posición
            location = env.hero.get_location()
            trajectory.append([location.x, location.y])
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Calcular errores respecto a ground truth
        ground_truth = interpolate_waypoints(route)
        
        for pos in trajectory:
            # Distancia a la ruta ideal
            min_dist = min([
                np.sqrt((pos[0]-gt[0])**2 + (pos[1]-gt[1])**2)
                for gt in ground_truth
            ])
            errors.append(min_dist)
            max_error = max(max_error, min_dist)
    
    # Calcular métricas
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    avg_time = np.mean(times)
    
    print(f"\n{'='*50}")
    print(f"Resultados de Validación ({num_iterations} iteraciones)")
    print(f"{'='*50}")
    print(f"RMSE: {rmse:.3f} m")
    print(f"Error Máximo: {max_error:.3f} m")
    print(f"Tiempo Promedio: {avg_time:.1f} s")
    print(f"{'='*50}")
    
    return {
        'rmse': rmse,
        'max_error': max_error,
        'avg_time': avg_time
    }
```

#### 4.2 Comparar con LQR (opcional)
```python
# Si tienes implementado un LQR controller
lqr_results = validate_agent(lqr_controller, route)
ddpg_results = validate_agent(ddpg_agent, route)

print(f"\nComparación:")
print(f"LQR:  RMSE={lqr_results['rmse']:.3f}m")
print(f"DDPG: RMSE={ddpg_results['rmse']:.3f}m")
print(f"Diferencia: {(ddpg_results['rmse']-lqr_results['rmse'])*100:.1f} cm")
```

---

## 🎯 PRIORIZACIÓN DE IMPLEMENTACIÓN

### ⭐⭐⭐ CRÍTICO (Hacer AHORA)
1. **Implementar DDPG** (Fase 2)
   - Mayor impacto
   - 50x más rápido
   - Mejor resultados

2. **Agregar Driving Features** (Fase 1.1)
   - Necesario para DDPG
   - Mejora estado actual
   - Fácil de implementar

3. **Mejorar Recompensa** (Fase 1.2)
   - Crucial para convergencia
   - Basada en paper validado
   - Impacto inmediato

### ⭐⭐ IMPORTANTE (Siguiente sprint)
4. **Agente Waypoints** (Fase 3)
   - Mejor del paper
   - Más simple que CNN
   - RMSE = 0.10m

5. **Agregar Sensores** (Fase 1.3)
   - Lane invasor
   - Collision detector
   - Terminar episodios correctamente

6. **Rutas Aleatorias** (Fase 1.4)
   - Generalización
   - Como paper
   - Evitar overfitting

### ⭐ OPCIONAL (Pulir después)
7. **Validación RMSE** (Fase 4)
   - Comparar resultados
   - Benchmarking
   - Paper-ready metrics

---

## 📈 RESULTADOS ESPERADOS

### Con Implementación Completa

| Métrica | Tu Proyecto Actual | Después de Mejoras | Meta (Paper) |
|---------|-------------------|-------------------|--------------|
| **Episodios DQN** | 10,000+ | N/A (migrar) | 8,300 |
| **Episodios DDPG** | N/A | 150-500 | 150 ⭐ |
| **RMSE** | ? | 0.10-0.15m | 0.10m ⭐ |
| **Tiempo Training** | Días | Horas | Horas ⭐ |
| **Control** | Discreto (sacudidas) | Continuo (suave) | Continuo ⭐ |
| **Transferible** | Difícil | Fácil | Fácil ⭐ |

---

## 🚀 PRÓXIMOS PASOS INMEDIATOS

1. **Leer** este documento completo
2. **Decidir**: ¿Migrar a DDPG o seguir con DQN?
3. **Implementar** Fase 1.1 + 1.2 (driving features + reward)
4. **Si eliges DDPG**: Implementar Fase 2
5. **Entrenar** y comparar resultados
6. **Validar** con métricas del paper (Fase 4)

---

## 💡 CONSEJO FINAL

El paper demostró que **DDPG-Carla-Waypoints** es la mejor opción:
- ✅ Más rápido (150 episodios vs 8,300)
- ✅ Mejor RMSE (0.10m vs 0.21m)
- ✅ Más simple (no necesita CNN)
- ✅ Control continuo (más realista)
- ✅ Transferible a vehículo real

**Recomendación**: Migrar a DDPG + Waypoints lo antes posible para ahorrar tiempo y recursos.

---

📝 **Documento actualizado**: Octubre 2025

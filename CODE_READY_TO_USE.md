# 🔧 Código Listo para Copiar/Pegar

Este archivo contiene código completo y listo para usar basado en el paper de referencia.

---

## 📦 1. DRIVING FEATURES (Agregar a base_env.py)

```python
def get_driving_features(self):
    """
    Extraer características de conducción como en el paper:
    vt: velocidad del vehículo (m/s)
    dt: distancia al centro del carril (m)
    φt: ángulo respecto al carril (radianes)
    """
    # 1. Velocidad del vehículo
    velocity = self.hero.get_velocity()
    vt = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s
    
    # 2. Waypoint más cercano (centro del carril)
    vehicle_location = self.hero.get_location()
    waypoint = self.world_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    # 3. Distancia al centro del carril
    lane_center = waypoint.transform.location
    dt = np.sqrt(
        (vehicle_location.x - lane_center.x)**2 + 
        (vehicle_location.y - lane_center.y)**2
    )
    
    # 4. Ángulo respecto al carril
    vehicle_yaw = self.hero.get_transform().rotation.yaw
    lane_yaw = waypoint.transform.rotation.yaw
    
    # Normalizar ángulo a [-180, 180]
    angle_diff = vehicle_yaw - lane_yaw
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    
    φt = np.radians(angle_diff)
    
    return np.array([vt, dt, φt], dtype=np.float32)
```

---

## 🎁 2. FUNCIÓN DE RECOMPENSA (Reemplazar en base_env.py)

```python
def calculate_reward(self):
    """
    Función de recompensa del paper:
    
    R = -200 si colisión o salida de carril
    R = Σ |vt·cos(φt)| - |vt·sin(φt)| - |vt|·|dt| si en carril
    R = +100 si meta alcanzada
    """
    # Obtener driving features
    vt, dt, φt = self.get_driving_features()
    
    # Verificar colisión
    if hasattr(self, 'collision_triggered') and self.collision_triggered:
        return -200.0
    
    # Verificar invasión de carril
    if hasattr(self, 'lane_invasion_triggered') and self.lane_invasion_triggered:
        return -200.0
    
    # Verificar si alcanzó la meta
    if self._is_goal_reached():
        return 100.0
    
    # Recompensa continua (ecuación del paper)
    reward = (
        np.abs(vt * np.cos(φt))      # Premia velocidad hacia adelante
        - np.abs(vt * np.sin(φt))     # Penaliza velocidad lateral (zigzag)
        - np.abs(vt) * np.abs(dt)     # Penaliza desviación del centro
    )
    
    return float(reward)

def _is_goal_reached(self):
    """Verificar si alcanzó el waypoint final"""
    if not hasattr(self, 'route_waypoints') or not self.route_waypoints:
        return False
    
    vehicle_location = self.hero.get_location()
    goal_location = self.route_waypoints[-1].transform.location
    
    distance = np.sqrt(
        (vehicle_location.x - goal_location.x)**2 + 
        (vehicle_location.y - goal_location.y)**2
    )
    
    return distance < 5.0  # 5 metros de tolerancia
```

---

## 🚨 3. SENSORES ADICIONALES (Agregar a carla_env.py)

```python
def setup_additional_sensors(self):
    """Agregar sensores de colisión y lane invasion como en el paper"""
    
    # Inicializar flags
    self.collision_triggered = False
    self.lane_invasion_triggered = False
    
    # 1. Sensor de colisión
    collision_bp = self.world.get_blueprint_library().find(
        'sensor.other.collision'
    )
    collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
    
    self.collision_sensor = self.world.spawn_actor(
        collision_bp,
        collision_transform,
        attach_to=self.hero
    )
    self.collision_sensor.listen(self._on_collision)
    
    # 2. Sensor de invasión de carril
    lane_invasion_bp = self.world.get_blueprint_library().find(
        'sensor.other.lane_invasion'
    )
    lane_invasion_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
    
    self.lane_invasion_sensor = self.world.spawn_actor(
        lane_invasion_bp,
        lane_invasion_transform,
        attach_to=self.hero
    )
    self.lane_invasion_sensor.listen(self._on_lane_invasion)
    
    print("✅ Sensores adicionales configurados (colisión + lane invasion)")

def _on_collision(self, event):
    """Callback cuando hay colisión"""
    self.collision_triggered = True
    other_actor = event.other_actor
    print(f"⚠️ Colisión detectada con {other_actor.type_id}")

def _on_lane_invasion(self, event):
    """Callback cuando invade carril"""
    # Solo penalizar salidas de carril significativas
    # Ignorar líneas discontinuas (permitidas)
    for marking in event.crossed_lane_markings:
        if marking.type in [
            carla.LaneMarkingType.Solid,
            carla.LaneMarkingType.SolidSolid
        ]:
            self.lane_invasion_triggered = True
            print(f"⚠️ Invasión de carril detectada: {marking.type}")
            break

def reset_sensors(self):
    """Resetear flags de sensores (llamar en reset())"""
    self.collision_triggered = False
    self.lane_invasion_triggered = False

def cleanup_sensors(self):
    """Limpiar sensores al destruir (llamar en __del__)"""
    if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
        self.collision_sensor.stop()
        self.collision_sensor.destroy()
    
    if hasattr(self, 'lane_invasion_sensor') and self.lane_invasion_sensor is not None:
        self.lane_invasion_sensor.stop()
        self.lane_invasion_sensor.destroy()
```

---

## 🗺️ 4. WAYPOINTS LOCALES (Agregar a base_env.py)

```python
def get_local_waypoints(self, num_waypoints=15, distance=2.0):
    """
    Obtener waypoints en coordenadas locales del vehículo
    Como en el paper: transformación a referencia local
    
    Args:
        num_waypoints: Número de waypoints a obtener (default: 15 como paper)
        distance: Distancia entre waypoints en metros (default: 2.0m)
    
    Returns:
        np.array: Array de shape (num_waypoints,) con coordenadas X locales
    """
    # Posición y orientación actual del vehículo
    vehicle_transform = self.hero.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    
    # Waypoint actual (proyectado a la carretera)
    current_waypoint = self.world_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    # Obtener próximos waypoints
    waypoints_global = []
    waypoint = current_waypoint
    
    for _ in range(num_waypoints):
        # Obtener siguiente waypoint a 'distance' metros
        next_waypoints = waypoint.next(distance)
        
        if not next_waypoints:
            # Si no hay más waypoints, usar el último
            break
            
        waypoint = next_waypoints[0]
        waypoints_global.append(waypoint)
    
    # Transformar a coordenadas locales del vehículo
    # Aplicar matriz de transformación del paper (rotación + traslación)
    waypoints_local = []
    
    yaw = np.radians(vehicle_rotation.yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    for wp in waypoints_global:
        # Trasladar (referenciar al vehículo)
        dx = wp.transform.location.x - vehicle_location.x
        dy = wp.transform.location.y - vehicle_location.y
        
        # Rotar (aplicar matriz de transformación)
        # x_local: coordenada lateral (izq/der del vehículo)
        # y_local: coordenada longitudinal (adelante/atrás)
        x_local = cos_yaw * dx + sin_yaw * dy
        y_local = -sin_yaw * dx + cos_yaw * dy
        
        # El paper usa solo la coordenada X (posición lateral)
        # porque indica cuánto girar para seguir el waypoint
        waypoints_local.append(x_local)
    
    # Pad con último valor si faltan waypoints
    # (puede pasar al final de la ruta)
    while len(waypoints_local) < num_waypoints:
        if waypoints_local:
            waypoints_local.append(waypoints_local[-1])
        else:
            waypoints_local.append(0.0)
    
    return np.array(waypoints_local[:num_waypoints], dtype=np.float32)
```

---

## 🔄 5. RUTAS ALEATORIAS (Modificar reset() en base_env.py)

```python
def reset(self, seed=None):
    """
    Reset con ruta aleatoria como en el paper
    Cada episodio usa una ruta diferente para generalización
    """
    # Resetear sensores
    if hasattr(self, 'collision_triggered'):
        self.reset_sensors()
    
    # Obtener puntos de spawn disponibles
    spawn_points = self.world_map.get_spawn_points()
    
    if len(spawn_points) < 2:
        raise ValueError("Mapa no tiene suficientes puntos de spawn")
    
    # Seleccionar inicio y fin aleatorios
    if seed is not None:
        np.random.seed(seed)
    
    start_idx = np.random.randint(0, len(spawn_points))
    end_idx = np.random.randint(0, len(spawn_points))
    
    # Asegurar que no sean el mismo punto
    max_attempts = 10
    attempts = 0
    while end_idx == start_idx and attempts < max_attempts:
        end_idx = np.random.randint(0, len(spawn_points))
        attempts += 1
    
    if end_idx == start_idx:
        # Si no encontramos punto diferente, usar el siguiente
        end_idx = (start_idx + 1) % len(spawn_points)
    
    start_location = spawn_points[start_idx].location
    end_location = spawn_points[end_idx].location
    
    # Calcular distancia de la ruta
    route_distance = np.sqrt(
        (end_location.x - start_location.x)**2 + 
        (end_location.y - start_location.y)**2
    )
    
    print(f"🗺️  Nueva ruta: Spawn {start_idx} → {end_idx} (~{route_distance:.0f}m)")
    
    # Generar ruta con A* planner de CARLA
    self.route_waypoints = self._get_route(start_location, end_location)
    
    if not self.route_waypoints:
        print("⚠️ No se pudo generar ruta, usando spawn aleatorio")
        # Fallback: solo spawn en punto inicial
        self.hero.set_transform(spawn_points[start_idx])
    else:
        # Spawn vehículo en punto inicial de la ruta
        start_transform = self.route_waypoints[0].transform
        self.hero.set_transform(start_transform)
        
        print(f"✅ Ruta generada: {len(self.route_waypoints)} waypoints")
    
    # Aplicar control inicial (vehículo quieto)
    self.hero.apply_control(carla.VehicleControl())
    
    # Esperar a que se estabilice
    self.world.tick()
    time.sleep(0.1)
    
    # Obtener observación inicial
    observation = self._get_observation()
    info = {'route_length': len(self.route_waypoints) if self.route_waypoints else 0}
    
    return observation, info

def _get_route(self, start_location, end_location):
    """
    Obtener ruta con A* planner de CARLA
    
    Returns:
        list: Lista de waypoints o None si falla
    """
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        
        # Crear planner (2.0m sampling resolution como en el paper)
        planner = GlobalRoutePlanner(self.world_map, 2.0)
        
        # Calcular ruta
        route = planner.trace_route(start_location, end_location)
        
        # Extraer solo waypoints (ignorar RoadOption)
        waypoints = [waypoint for waypoint, _ in route]
        
        return waypoints
        
    except Exception as e:
        print(f"❌ Error generando ruta: {e}")
        return None
```

---

## 🧪 6. VALIDACIÓN RMSE (Nuevo archivo: src/utils/validation.py)

```python
"""
Validación de agentes como en el paper
Calcular RMSE comparando trayectoria con ground truth
"""

import numpy as np
import time
from scipy.interpolate import interp1d

def interpolate_route(waypoints, num_points=1000):
    """
    Interpolar ruta de waypoints para obtener ground truth suave
    
    Args:
        waypoints: Lista de carla.Waypoint
        num_points: Número de puntos interpolados
    
    Returns:
        np.array: Array (num_points, 2) con coordenadas [x, y]
    """
    if not waypoints or len(waypoints) < 2:
        return np.array([])
    
    # Extraer coordenadas
    xs = [wp.transform.location.x for wp in waypoints]
    ys = [wp.transform.location.y for wp in waypoints]
    
    # Calcular distancia acumulada
    distances = [0]
    for i in range(1, len(xs)):
        dist = np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2)
        distances.append(distances[-1] + dist)
    
    # Interpolar
    if distances[-1] == 0:
        return np.array([[xs[0], ys[0]]])
    
    fx = interp1d(distances, xs, kind='linear', fill_value='extrapolate')
    fy = interp1d(distances, ys, kind='linear', fill_value='extrapolate')
    
    # Puntos uniformemente espaciados
    s_new = np.linspace(0, distances[-1], num_points)
    xs_new = fx(s_new)
    ys_new = fy(s_new)
    
    return np.column_stack([xs_new, ys_new])

def calculate_trajectory_error(trajectory, ground_truth):
    """
    Calcular error entre trayectoria y ground truth
    
    Args:
        trajectory: List of [x, y] positions
        ground_truth: np.array (N, 2) de ground truth
    
    Returns:
        tuple: (errores, error_máximo)
    """
    errors = []
    
    for pos in trajectory:
        # Distancia mínima a cualquier punto del ground truth
        distances = np.sqrt(
            (ground_truth[:, 0] - pos[0])**2 + 
            (ground_truth[:, 1] - pos[1])**2
        )
        min_dist = np.min(distances)
        errors.append(min_dist)
    
    max_error = np.max(errors) if errors else 0.0
    
    return errors, max_error

def validate_agent(env, agent, route_waypoints, num_iterations=20, noise=0.0):
    """
    Validar agente como en el paper:
    - Misma ruta múltiples veces
    - Comparar con ground truth
    - Calcular RMSE, error máximo, tiempo
    
    Args:
        env: Entorno CARLA
        agent: Agente DRL (debe tener método select_action)
        route_waypoints: Lista de waypoints de la ruta
        num_iterations: Número de iteraciones (default: 20 como paper)
        noise: Ruido para exploración (default: 0.0 = sin ruido)
    
    Returns:
        dict: Métricas {rmse, max_error, avg_time, success_rate}
    """
    print(f"\n{'='*60}")
    print(f"🧪 VALIDACIÓN DEL AGENTE")
    print(f"{'='*60}")
    print(f"Ruta: {len(route_waypoints)} waypoints")
    print(f"Iteraciones: {num_iterations}")
    print(f"{'='*60}\n")
    
    # Interpolar ground truth
    ground_truth = interpolate_route(route_waypoints)
    
    all_errors = []
    times = []
    max_errors = []
    successes = 0
    
    for i in range(num_iterations):
        print(f"📍 Iteración {i+1}/{num_iterations}...", end=" ")
        
        # Reset con ruta específica
        env.reset_with_route(route_waypoints)
        
        trajectory = []
        start_time = time.time()
        done = False
        truncated = False
        steps = 0
        max_steps = 10000  # Timeout
        
        while not done and not truncated and steps < max_steps:
            # Obtener estado
            state = env.get_state()
            
            # Seleccionar acción (sin ruido para evaluación)
            action = agent.select_action(state, noise=noise)
            
            # Ejecutar acción
            next_state, reward, done, truncated, info = env.step(action)
            
            # Guardar posición
            location = env.hero.get_location()
            trajectory.append([location.x, location.y])
            
            steps += 1
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Verificar si completó la ruta
        if info.get('reached_goal', False):
            successes += 1
            success_str = "✅"
        else:
            success_str = "❌"
        
        # Calcular errores
        errors, max_error = calculate_trajectory_error(trajectory, ground_truth)
        all_errors.extend(errors)
        max_errors.append(max_error)
        
        print(f"{success_str} {elapsed:.1f}s, max_err={max_error:.3f}m")
    
    # Calcular métricas finales
    rmse = np.sqrt(np.mean(np.array(all_errors)**2))
    max_error_overall = np.max(max_errors)
    avg_time = np.mean(times)
    success_rate = (successes / num_iterations) * 100
    
    # Imprimir resultados
    print(f"\n{'='*60}")
    print(f"📊 RESULTADOS DE VALIDACIÓN")
    print(f"{'='*60}")
    print(f"RMSE:              {rmse:.3f} m")
    print(f"Error Máximo:      {max_error_overall:.3f} m")
    print(f"Tiempo Promedio:   {avg_time:.1f} s")
    print(f"Tasa de Éxito:     {success_rate:.1f}% ({successes}/{num_iterations})")
    print(f"{'='*60}\n")
    
    # Comparación con paper
    print(f"📖 Comparación con Paper (Pérez-Gil et al. 2022):")
    print(f"   LQR Controller:        RMSE = 0.095 m")
    print(f"   DDPG-Waypoints:        RMSE = 0.10 m  ⭐ Mejor DRL")
    print(f"   DDPG-Pre-CNN:          RMSE = 0.115 m")
    print(f"   DQN-Waypoints:         RMSE = 0.21 m")
    print(f"   Tu Agente:             RMSE = {rmse:.3f} m")
    
    if rmse <= 0.10:
        print(f"   🏆 ¡EXCELENTE! Superaste el mejor agente del paper!")
    elif rmse <= 0.15:
        print(f"   ✅ ¡MUY BIEN! Resultados comparables al paper")
    elif rmse <= 0.25:
        print(f"   ⚠️  Resultados aceptables, hay margen de mejora")
    else:
        print(f"   ❌ Necesita más entrenamiento o ajustes")
    
    print(f"{'='*60}\n")
    
    return {
        'rmse': rmse,
        'max_error': max_error_overall,
        'avg_time': avg_time,
        'success_rate': success_rate,
        'all_errors': all_errors,
        'times': times
    }

def compare_agents(env, agents_dict, route_waypoints, num_iterations=20):
    """
    Comparar múltiples agentes
    
    Args:
        env: Entorno CARLA
        agents_dict: Dict {'nombre': agente}
        route_waypoints: Ruta para validar
        num_iterations: Iteraciones por agente
    
    Returns:
        dict: Resultados por agente
    """
    results = {}
    
    for name, agent in agents_dict.items():
        print(f"\n🤖 Validando: {name}")
        results[name] = validate_agent(env, agent, route_waypoints, num_iterations)
    
    # Tabla comparativa
    print(f"\n{'='*80}")
    print(f"📊 COMPARACIÓN DE AGENTES")
    print(f"{'='*80}")
    print(f"{'Agente':<25} {'RMSE (m)':<12} {'Error Máx (m)':<15} {'Tiempo (s)':<12}")
    print(f"{'-'*80}")
    
    for name, res in results.items():
        print(f"{name:<25} {res['rmse']:<12.3f} {res['max_error']:<15.3f} {res['avg_time']:<12.1f}")
    
    print(f"{'='*80}\n")
    
    return results
```

---

## 🚀 7. EJEMPLO DE USO COMPLETO

```python
"""
Ejemplo completo de uso de todas las mejoras
"""

# 1. En main.py - Entrenar con mejoras
def train_with_improvements():
    import gymnasium as gym
    from src.agents.ddpg_agent import DDPGAgent
    
    # Crear entorno
    env = gym.make('CarlaEnv-v0')
    
    # Crear agente DDPG
    agent = DDPGAgent(
        state_dim=(4, 84, 84),  # Frame stack
        action_dim=2,  # [steering, throttle]
        lr_actor=1e-4,
        lr_critic=1e-3
    )
    
    num_episodes = 500  # Como en el paper para DDPG
    
    for episode in range(num_episodes):
        # Reset con ruta aleatoria (automático)
        state, info = env.reset()
        
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        
        while not done and not truncated:
            # Seleccionar acción
            action = agent.select_action(state, noise=0.1)
            
            # Ejecutar
            next_state, reward, done, truncated, info = env.step(action)
            
            # Guardar en replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Entrenar
            if len(agent.replay_buffer) > agent.batch_size:
                agent.train()
            
            state = next_state
            episode_reward += reward
            step += 1
        
        print(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={step}")
        
        # Guardar mejor modelo
        if episode_reward > best_reward:
            agent.save(f"models/ddpg_best_{episode}.pth")
            best_reward = episode_reward

# 2. Validar agente entrenado
def validate_trained_agent():
    from src.utils.validation import validate_agent, compare_agents
    
    # Cargar agente
    agent = DDPGAgent.load("models/ddpg_best_150.pth")
    
    # Cargar ruta de prueba (180m como en paper)
    route = get_test_route()  # Implementar según tu setup
    
    # Validar
    results = validate_agent(
        env=env,
        agent=agent,
        route_waypoints=route,
        num_iterations=20
    )
    
    # Resultados automáticos vs paper
    print(f"Tu RMSE: {results['rmse']:.3f}m")
    print(f"Paper DDPG-Waypoints: 0.10m")

if __name__ == "__main__":
    # Modo 1: Entrenar
    # train_with_improvements()
    
    # Modo 2: Validar
    validate_trained_agent()
```

---

## ✅ CHECKLIST DE IMPLEMENTACIÓN

Marca cuando completes cada parte:

### Fase 1: Mejoras Inmediatas
- [ ] Agregar `get_driving_features()` a base_env.py
- [ ] Implementar nueva `calculate_reward()` con fórmula del paper
- [ ] Agregar `setup_additional_sensors()` (colisión + lane invasion)
- [ ] Modificar `reset()` para rutas aleatorias
- [ ] Agregar `get_local_waypoints()` para futuro uso

### Fase 2: DDPG (Opcional pero recomendado)
- [ ] Crear src/agents/ddpg_agent.py
- [ ] Implementar clase Actor (red neuronal)
- [ ] Implementar clase Critic (red neuronal)
- [ ] Implementar clase DDPGAgent (completa)
- [ ] Probar con episodio de prueba

### Fase 3: Validación
- [ ] Crear src/utils/validation.py
- [ ] Implementar `interpolate_route()`
- [ ] Implementar `calculate_trajectory_error()`
- [ ] Implementar `validate_agent()`
- [ ] Implementar `compare_agents()`
- [ ] Crear ruta de prueba fija
- [ ] Ejecutar validación con 20 iteraciones

### Fase 4: Documentación
- [ ] Registrar RMSE obtenido
- [ ] Comparar con resultados del paper
- [ ] Documentar cambios en README
- [ ] Crear gráficas de resultados

---

📝 **Última actualización**: Octubre 2025  
🎯 **Objetivo**: RMSE < 0.15m (comparable a DDPG del paper)

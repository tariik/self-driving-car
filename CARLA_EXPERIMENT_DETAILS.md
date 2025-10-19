# 📋 Configuración CARLA del Paper de Referencia

**Paper**: "Deep reinforcement learning based control for Autonomous Vehicles in CARLA"  
**Autores**: Óscar Pérez-Gil et al. (Universidad de Alcalá, 2022)  
**DOI**: 10.1007/s11042-021-11437-3

---

## 🎯 OBJETIVO DEL EXPERIMENTO

Entrenar agentes DRL (DQN y DDPG) para **navegación autónoma** en CARLA:
- Seguir una ruta predeterminada lo más rápido posible
- Evitar colisiones y salidas de carril
- Navegar en entorno urbano dinámico

---

## 🖥️ CONFIGURACIÓN TÉCNICA

### Hardware Utilizado
```
CPU: Intel Core i7-9700k
RAM: 32GB
GPU: NVIDIA GeForce RTX 2080 Ti (11GB VRAM)
CUDA: Habilitado
```

### Software
- **CARLA Simulator**: Basado en Unreal Engine 4
- **Framework**: Python API de CARLA
- **RL Frameworks**: Múltiples frameworks open-source
- **Integración**: Posible con ROS (CARLA ROS Bridge)

---

## 🗺️ MAPAS Y RUTAS

### Mapas Utilizados
- **Town01**: Principal mapa de validación
  - Ruta de validación: ~180 metros
  - Incluye curvas en ambas direcciones
  - Incluye secciones rectas

### Sistema de Rutas
- **Planificador global**: A* basado en waypoints
- **Generación dinámica**: Rutas aleatorias en cada episodio
- **Fuente**: Waypoints provistos por CARLA PythonAPI
- **Formato**: OpenDrive standard para definir carreteras

---

## 🎮 ESPACIO DE ESTADOS (State Space)

El estado `S` se modela como tupla `st = (vft, dft)`:

### 1. Visual Features Vector (vft)
Depende del agente específico (ver sección "Agentes")

### 2. Driving Features Vector (dft)
```python
dft = (vt, dt, φt)
```
- **vt**: Velocidad del vehículo (m/s)
- **dt**: Distancia al centro del carril (m)
- **φt**: Ángulo respecto al centro del carril (radianes)

**Fuente**: Estos datos son provistos directamente por CARLA Simulator

---

## 🕹️ ESPACIO DE ACCIONES (Action Space)

### DQN (Discreto)
- **Total acciones**: 27 comandos discretos
- **Steering**: 9 posiciones [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
- **Throttle**: 3 posiciones [0, 0.5, 1]
- **Brake**: No implementado (freno regenerativo del vehículo es suficiente)

### DDPG (Continuo) ⭐ MEJOR
- **Steering**: Continuo [-1, 1]
- **Throttle**: Continuo [0, 1]
- **Brake**: No implementado
- **Ventaja**: Control más suave y realista

---

## 🎁 FUNCIÓN DE RECOMPENSA (Reward Function)

```python
# Colisión, cambio de carril o salida de carretera
R = -200

# Mientras el carro está en el carril
R = Σ |vt·cos(φt)| - |vt·sin(φt)| - |vt|·|dt|

# Meta alcanzada
R = +100
```

### Componentes de la Recompensa
- **+|vt·cos(φt)|**: Premia velocidad longitudinal (avanzar)
- **-|vt·sin(φt)|**: Penaliza velocidad transversal (zigzaguear)
- **-|vt|·|dt|**: Penaliza desviación del centro del carril
- **-200**: Penalización fuerte por colisión/salida
- **+100**: Recompensa por completar ruta

---

## 📷 SENSORES UTILIZADOS

### 1. Cámara RGB
- **Resolución original**: 640x480 píxeles
- **Procesamiento**: Varía según agente (ver "Agentes")
- **Ubicación**: Vista frontal del vehículo
- **Uso**: Extracción de características visuales de la carretera

### 2. Sensor de Invasión de Carril (Lane Invasor)
- **Propósito**: Detectar salidas de carril
- **Acción**: Termina episodio si se activa

### 3. Sensor de Colisión (Collision Sensor)
- **Propósito**: Detectar colisiones
- **Acción**: Termina episodio si se activa

### 4. Waypoints
- **Fuente**: A* global planner de CARLA
- **Formato**: Lista de puntos (x, y, z) en coordenadas globales
- **Transformación**: Convertidos a coordenadas locales del vehículo

```python
# Matriz de transformación (rotación + traslación)
M = [[cos(φc), -sin(φc), 0, Xc],
     [sin(φc),  cos(φc), 0, Yc],
     [0,        0,       1, Zc],
     [0,        0,       0, 1]]
```
- **(Xc, Yc, Zc)**: Posición global del vehículo
- **φc**: Heading/yaw del vehículo

---

## 🤖 AGENTES PROPUESTOS (4 Arquitecturas)

### 1. DRL-Flatten-Image Agent
**Entrada**: Imagen B/N segmentada de la carretera
- **Procesamiento**: 640x480 → 11x11 (reducción de 300k a 121 datos)
- **Flatten**: Vector de 121 componentes
- **Estado**: `S = ([P₀, P₁, ..., P₁₂₀], φt, dt)`
- **Red**: 2 capas Fully-Connected
- **Ventaja**: Muy simple y rápido

### 2. DRL-Carla-Waypoints Agent ⭐ MEJOR DDPG
**Entrada**: Waypoints directamente de CARLA
- **Cantidad**: 15 waypoints (ventana FIFO)
- **Referencia**: Transformados a coordenadas locales del vehículo
- **Coordenadas usadas**: Solo X (posición lateral en el carril)
- **Estado**: `S = ([wp₀...wp₁₄], φt, dt)`
- **Red**: 2 capas Fully-Connected
- **RMSE**: 0.10m (mejor resultado DDPG)

### 3. DRL-CNN Agent
**Entrada**: Imagen RGB 640x480
- **CNN**: 3 capas convolucionales
  - Conv1: 64 filtros [7x7] + ReLU + AvgPool
  - Conv2: 64 filtros [5x5] + ReLU + AvgPool
  - Conv3: 64 filtros [3x3] + ReLU + AvgPool
- **Flatten**: Salida CNN aplanada
- **Estado**: `S = ([It], φt, dt)`
- **Red**: CNN + 2 capas Fully-Connected
- **Desventaja**: 307,200 datos → entrenamiento muy lento

### 4. DRL-Pre-CNN Agent
**Entrada**: Imagen RGB 640x480
- **CNN Pre-entrenado**: Offline con dataset de imágenes-waypoints
- **Función**: Predecir waypoints desde imagen
- **Estado real**: `S = ([wp₀...wp₁₄], φt, dt)` (waypoints predichos)
- **Red**: CNN (pre-entrenado) + 2 capas Fully-Connected
- **RMSE DDPG**: 0.115m (segundo mejor)

---

## 🏋️ PROCESO DE ENTRENAMIENTO

### Workflow de Entrenamiento

```python
for episode in range(M_episodes):
    # 1. Inicializar episodio
    route = a_star_planner.get_random_route()  # Ruta aleatoria
    
    for step in range(T_steps):
        # 2. Observar estado
        state = get_observation()  # S = ([D], φt, dt)
        
        # 3. Predecir acción
        action = drl_agent.predict(state)  # (throttle, steering)
        
        # 4. Ejecutar acción en CARLA
        carla.apply_control(action)
        
        # 5. Calcular recompensa
        reward = calculate_reward()
        
        # 6. Verificar sensores
        if lane_invasor.triggered or collision_sensor.triggered:
            episode.end()
            vehicle.reset()  # Reposicionar en centro del carril
            break
        
        # 7. Entrenar agente
        drl_agent.train(state, action, reward, next_state)
```

### Parámetros de Entrenamiento

#### DQN
- **Episodios totales**: 20,000 - 120,000 (según agente)
- **Mejor episodio**: 8,300 - 108,600
- **Criterio**: Mayor recompensa acumulada + mayor distancia recorrida
- **Tiempo**: Mucho más largo que DDPG

#### DDPG ⭐ MEJOR
- **Episodios totales**: 500 - 60,000 (según agente)
- **Mejor episodio**: 50 - 45,950
- **Convergencia**: Modelos tempranos (< 200 episodios)
- **Tiempo**: Drásticamente reducido vs DQN

---

## 📊 VALIDACIÓN Y MÉTRICAS

### Proceso de Validación

1. **Ruta fija**: Seleccionar ruta específica en el mapa
2. **Iteraciones**: Conducir 20 veces la misma ruta
3. **Ground Truth**: Ruta ideal obtenida interpolando waypoints de CARLA
4. **Comparación**: vs LQR controller (método clásico)

### Métricas Calculadas

#### RMSE (Root Mean Square Error)
```python
RMSE = √(Σ(posición_real - posición_ideal)² / N)
```

#### Error Máximo
```python
max_error = max(|posición_real - posición_ideal|)
```

#### Tiempo Promedio
Tiempo promedio en completar la ruta

### Resultados Town01 (180m)

| Método | RMSE (m) | Error Máx (m) | Tiempo (s) |
|--------|----------|---------------|------------|
| **LQR (Clásico)** | 0.06 | 0.74 | 17.4 |
| Manual Control | 0.40 | 1.80 | 22.7 |
| **DDPG-Waypoints** ⭐ | **0.13** | **1.50** | **20.6** |
| **DDPG-Pre-CNN** | **0.10** | **1.41** | **23.8** |
| DDPG-Flatten | 0.15 | 1.43 | 19.9 |
| DDPG-CNN | 0.75 | 2.55 | 34.2 |
| DQN-Waypoints | 0.21 | 1.32 | 29.3 |
| DQN-Flatten | 0.64 | 3.15 | 27.3 |
| DQN-CNN | 0.83 | 2.15 | 33.3 |

### Resultados 20 Rutas Variadas (180-700m)

| Método | RMSE (m) | Error Máx (m) | Tiempo (s) |
|--------|----------|---------------|------------|
| **LQR (Clásico)** | 0.095 | 1.305 | 65.60 |
| **DDPG-Waypoints** ⭐ | **0.10** | **1.46** | **62.25** |
| **DDPG-Pre-CNN** | **0.115** | **1.512** | **65.12** |
| DDPG-Flatten | 0.134 | 1.522 | 63.97 |
| DDPG-CNN | 0.67 | 2.78 | 125.43 |

---

## 🔑 CONCLUSIONES CLAVE DEL PAPER

### ✅ DDPG vs DQN
- **DDPG es SUPERIOR**: Control continuo más realista
- **DQN limitado**: Naturaleza discreta dificulta el control suave
- **Reducción de tiempo**: DDPG entrena ~50x más rápido

### ⭐ Mejor Agente: DDPG-Carla-Waypoints
- **RMSE**: 0.10m (prácticamente igual a LQR: 0.095m)
- **Ventaja sobre clásicos**: No requiere tuning manual complejo
- **Reproducibilidad**: Fácil replicación en cualquier entorno
- **Sin conocimiento experto**: No necesita teoría de control electrónico

### 🎯 Agentes Recomendados (en orden)
1. **DDPG-Carla-Waypoints**: Mejor performance (RMSE 0.10m)
2. **DDPG-Pre-CNN**: Segundo mejor (RMSE 0.115m)
3. **DDPG-Flatten-Image**: Balance velocidad/precisión (RMSE 0.134m)
4. **DQN-Waypoints**: Si necesitas acciones discretas (RMSE 0.21m)

### ⚠️ Evitar
- **DRL-CNN** (ambos algoritmos): Muy lento, peores resultados

---

## 🚀 CARACTERÍSTICAS CARLA UTILIZADAS

### Ventajas de CARLA para este proyecto

1. **Seguridad**: Pruebas ilimitadas sin riesgo
2. **Ground Truth**: Odometría real + ruta ideal disponibles
3. **Flexibilidad**: Control total de clima, peatones, tráfico, sensores
4. **Fast Simulation**: Modo sin render para entrenamiento rápido
5. **Realismo**: Apariencia hiper-realista (Unreal Engine 4)
6. **Escenarios**: Scenario Runner para construir casos de prueba
7. **API Python**: Control programático completo
8. **Gratuito**: Open-source

### Configuración CARLA Utilizada

```python
# Modo de ejecución
- Fast simulation: Render deshabilitado para entrenamiento
- Normal mode: Con render para validación visual

# Control de simulación
- Physics: GPU dedicada recomendada
- Client-side: Control de lógica de actores
- Server-side: Cálculo de física

# Integración
- Python API: Comunicación directa
- ROS Bridge: Integración futura con vehículo real
```

---

## 📝 APLICACIÓN A TU PROYECTO

### Tu Configuración Actual (Recomendaciones)

```python
# ✅ BIEN - Configuraciones que coinciden con el paper
- GPU: RTX 3080 (10GB) - Similar a RTX 2080 Ti (11GB)
- CUDA: Habilitado
- Framework: Python + Gymnasium
- Algoritmo: DQN (considerado en el paper)

# ⚠️ DIFERENCIAS - Ajustes recomendados
- Cámara: Actualmente frontal (1.5, 0, 1.5, 0°, 0°, 0°) ✅ CORRECTO
- Resolución: 84x84 grayscale - Paper usa 640x480 RGB
- Estado: Frame stack (4 frames) - Paper usa características de conducción
- Acciones: 29 clases discretas - Paper usa 27 clases DQN

# 🎯 RECOMENDACIÓN: Implementar DDPG
- Mejor performance demostrada
- Entrenamiento más rápido
- Control continuo más realista
- Tranferencia a vehículo real más fácil
```

### Mejoras Sugeridas Basadas en el Paper

1. **Algoritmo**: Cambiar de DQN a DDPG
2. **Estado**: Agregar driving features (vt, dt, φt) de CARLA
3. **Waypoints**: Usar planificador A* de CARLA (como DDPG-Waypoints)
4. **Sensores**: Agregar lane_invasor y collision_sensor
5. **Recompensa**: Implementar función de recompensa del paper
6. **Validación**: Comparar con ground truth de waypoints de CARLA
7. **Rutas**: Generar rutas aleatorias en cada episodio

---

## 📚 REFERENCIAS

- Paper original: https://doi.org/10.1007/s11042-021-11437-3
- CARLA Simulator: https://carla.org
- Unreal Engine 4: https://www.unrealengine.com
- OpenDrive: https://www.asam.net/standards/detail/opendrive/

---

**Fecha de análisis**: Octubre 2025  
**Preparado para**: Proyecto self-driving-car CARLA DRL

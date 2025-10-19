# 📊 TensorBoard Integration - DRL-Flatten-Image

## Métricas Implementadas

Basado en el paper Pérez-Gil et al. (2022), el sistema de logging con TensorBoard incluye:

### 1. Métricas por Episodio (`Episode/*`)
- **Total_Reward**: Reward acumulado en el episodio
- **Length**: Número de steps en el episodio
- **Avg_Reward_Per_Step**: Reward promedio por step
- **Collision**: Si hubo colisión (binario: 0 o 1)
- **Lane_Invasion**: Si hubo invasión de carril (binario: 0 o 1)

### 2. Métricas Acumuladas (`Cumulative/*`)
- **Collision_Rate**: Tasa de colisiones a lo largo del entrenamiento
- **Lane_Invasion_Rate**: Tasa de invasiones de carril

### 3. Métricas de Entrenamiento (`Training/*`)
- **Epsilon**: Tasa de exploración (epsilon-greedy para DQN)
- **Loss**: Loss de la red neuronal (si disponible)

### 4. Estado del Vehículo (`State/*`)
Según el paper (dft = (vt, dt, φt)):
- **Velocity_vt**: Velocidad del vehículo (m/s)
- **Distance_dt**: Distancia al centro del carril (m)
- **Angle_phi_t**: Ángulo con respecto al carril (grados)

### 5. Acciones de Control (`Action/*`)
- **Throttle**: Aceleración [0, 1]
- **Steering**: Dirección [-1, 1]
- **Brake**: Frenado [0, 1]

### 6. Reward por Step (`Step/*`)
- **Reward**: Reward instantáneo en cada step

### 7. Promedios Móviles (`Running_Avg_N/*`)
- **Reward**: Promedio móvil de rewards (ventana configurable)
- **Length**: Promedio móvil de longitudes de episodio

### 8. Mejor Modelo (`Best/*`)
- **Episode**: Episodio del mejor modelo
- **Reward**: Mejor reward alcanzado

## Uso

### Iniciar TensorBoard

```bash
# Desde el directorio del proyecto
tensorboard --logdir=runs

# Luego abrir en navegador: http://localhost:6006
```

### Configuración en Código

El logger se inicializa automáticamente en `main.py`:

```python
tb_logger = TensorBoardLogger(log_dir='runs', experiment_name=experiment_name)
```

### Hiperparámetros Logged

Todos los hiperparámetros del experimento se guardan automáticamente:
- Algoritmo (DQN/DDPG)
- Arquitectura (DRL-Flatten-Image)
- Buffer size, batch size, gamma, learning rate
- Epsilon parameters
- State/action sizes
- Image resolution

## Visualizaciones Disponibles

### 1. Scalars
- Gráficas de todas las métricas numéricas a lo largo del entrenamiento
- Permite comparar múltiples runs

### 2. Histograms
- Distribución de valores (si se loggean histogramas)

### 3. HParams
- Comparación de hiperparámetros entre diferentes experimentos
- Búsqueda de mejores configuraciones

## Comparación con Paper

El paper Pérez-Gil et al. (2022) usa estas métricas principales:
1. ✅ **Total reward** por episodio
2. ✅ **Episode length** (número de steps)
3. ✅ **Driving features** (vt, dt, φt)
4. ✅ **Collision/Lane invasion** events
5. ✅ **RMSE** en validación (calculado offline)

Todas estas métricas están implementadas y se visualizan en TensorBoard.

## Frecuencia de Logging

Para evitar overhead excesivo:
- **Métricas por step**: Cada 10 steps (configurable)
- **Métricas por episodio**: Cada episodio
- **Checkpoints**: Cada 50 episodios (configurable)

## Ejemplo de Salida

```
📊 TensorBoard logger initialized: runs/DQN_Flatten_20251019_143025
   Run: tensorboard --logdir=runs
```

## Notas

- Los logs se guardan en `runs/<experiment_name>/`
- Cada experimento tiene un timestamp único
- Los hiperparámetros se guardan para reproducibilidad
- Compatible con múltiples runs paralelos

## Referencias

Paper: Pérez-Gil et al. (2022) "Deep reinforcement learning based control for Autonomous Vehicles in CARLA"
- DOI: 10.1007/s11042-021-11437-3
- Sección 6: Results (métricas de evaluación)

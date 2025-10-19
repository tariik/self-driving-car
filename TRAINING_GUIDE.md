# 🚗 DRL-Flatten-Image: Entrenamiento y Evaluación

## 📚 Implementación del Paper
**"Deep reinforcement learning based control for Autonomous Vehicles in CARLA"**  
Pérez-Gil et al. (2022) - Sección 5.1: DRL-Flatten-Image Agent

## 🎯 Arquitectura Implementada

### Estado (Ecuación 21)
```
S = ([Pt0, Pt1, ..., Pt120], φt, dt)
```
- **Imagen**: 11×11 pixels aplanada = 121 valores
- **φt**: Ángulo al carril (radianes)
- **dt**: Distancia al centro (metros)
- **Total**: 123 dimensiones

### Red Neuronal (DQN)
```
Input (123) → FC1 (64, ReLU) → FC2 (32, ReLU) → Output (27 acciones)
```

### Acciones Discretas (Tabla 1 del paper)
- **Steering**: 9 posiciones [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
- **Throttle**: 3 posiciones [0, 0.5, 1]
- **Total**: 9 × 3 = 27 combinaciones

## 🚀 Uso

### 1️⃣ Entrenamiento
```bash
# Entrenar modelo nuevo (500 episodios por defecto)
python src/main.py

# Checkpoints se guardan cada 50 episodios en:
# checkpoints/drl_flatten_episode_50.pth
# checkpoints/drl_flatten_episode_100.pth
# ...
# checkpoints/drl_flatten_final.pth
```

**Hiperparámetros:**
- Episodios: 500 (paper: DDPG necesita ~500, DQN ~20,000)
- Steps por episodio: 500
- Learning rate: 5e-4
- Batch size: 32
- Gamma: 0.99
- Replay buffer: 100,000

### 2️⃣ Evaluación
```bash
# Evaluar último checkpoint (20 iteraciones)
python evaluate_agent.py

# Evaluar checkpoint específico
python evaluate_agent.py --checkpoint checkpoints/drl_flatten_episode_200.pth

# Evaluar con más iteraciones
python evaluate_agent.py --iterations 50

# Ver guía completa
cat EVALUATION_GUIDE.md
```

**Métricas calculadas:**
- RMSE (Root Mean Square Error)
- Error máximo
- Tasa de éxito
- Tiempo promedio
- Comparación con paper

### 3️⃣ Monitoreo Durante Training

**Console output:**
```
Episode 1/500
   ✓ Episode finished at step 245: Total reward = 123.45
   🏆 NEW BEST! Episode 1: Reward = 123.45

Episodes 1-10 Summary:
   Avg Reward: 98.32
   Avg Length: 234.5 steps
   Best so far: Episode 3 with 145.67

💾 Checkpoint saved: checkpoints/drl_flatten_episode_50.pth
📊 Avg reward (last 50): 156.78
```

## 📊 Resultados Esperados (Paper - Tabla 3)

| Método                 | RMSE (m) | Training Episodes |
|------------------------|----------|-------------------|
| LQR (baseline)         | 0.06     | N/A               |
| DQN-Flatten-Image      | 0.08     | ~16,500           |
| DDPG-Flatten-Image     | 0.07     | ~50               |

**Tu objetivo**: RMSE < 0.10 m (excelente), < 0.15 m (bueno)

## 🛠️ Configuración del Entorno

### Sensores (Ecuación 21)
- **Cámara RGB**: 640×480 → resize a 11×11 → grayscale → normalizada [-1,1]
- **Posición**: x=2.0m adelante, z=1.2m altura
- **Orientación**: pitch=-25° (mirando a la carretera)
- **Driving features**: φt (ángulo) y dt (distancia) desde CARLA API

### Mapa y Rutas
- **Mapa**: Town01 (del paper)
- **Clima**: ClearNoon
- **Rutas**: Aleatorias en cada episodio (A* planner)
- **Visualización**: Waypoints verdes a 5m altura

### Función de Recompensa (Paper)
```python
R = -200  # si colisión o invasión de carril
R = Σ |vt·cos(φt)| - |vt·sin(φt)| - |vt|·|dt|  # si en carril
R = +100  # si meta alcanzada
```

## 📁 Estructura de Archivos

```
self-driving-car/
├── src/
│   ├── main.py                    # Training loop
│   ├── agents/
│   │   └── drl_flatten_agent.py   # DQN implementation
│   ├── env/
│   │   ├── base_env.py            # Environment base
│   │   └── carla_env.py           # CARLA wrapper
│   └── utils/
│       ├── video_recorder.py      # Video recording
│       └── display_manager.py     # Visualization
├── evaluate_agent.py              # Evaluation script
├── EVALUATION_GUIDE.md            # Evaluation guide
├── checkpoints/                   # Saved models
├── evaluation_output/             # Evaluation videos
└── render_output/                 # Training videos (disabled)
```

## 🎮 Controles

### Durante Training
- **ESC** o **Q**: Salir
- **Ventana**: Muestra vista chase-cam detrás del vehículo

### Durante Evaluation
- **ESC** o **Q**: Salir y guardar resultados parciales

## 🐛 Troubleshooting

### CARLA no está corriendo
```bash
./launch_carla.sh
# Esperar a "Listening on port 3000"
```

### GPU Out of Memory
El código automáticamente fallback a CPU. Para forzar CPU:
```python
device = torch.device("cpu")
```

### Display no funciona (headless)
```bash
python src/main.py  # Training no necesita display
python evaluate_agent.py --no-display  # Evaluation headless
```

### Training muy lento
1. Desactiva renders (ya desactivado por defecto)
2. Reduce batch size: `BATCH_SIZE = 16`
3. Reduce episodios: `MAX_EPISODES = 100`

## 📈 Mejoras Sugeridas

### Para mejor RMSE
1. Más episodios (paper DQN usa ~20K)
2. Ajustar learning rate
3. Usar DDPG en vez de DQN (mejor para control continuo)

### Para mejor velocidad
1. Añadir término de velocidad a recompensa
2. Penalizar más la lentitud
3. Usar throttle variable

### Para mejor estabilidad
1. Aumentar penalty de colisión
2. Añadir penalty gradual por desviación
3. Usar soft update más suave (TAU más pequeño)

## 📚 Archivos de Configuración

### paper_text.txt
Texto completo del paper original

### PHASE_1_COMPLETED.md
Documentación de implementación fase 1

### DRL_FLATTEN_IMAGE_PAPER_CONFIG.md
Detalles de configuración del paper

## 🎓 Referencias

**Paper Original:**
Pérez-Gil, Ó., Barea, R., López-Guillén, E., Bergasa, L. M., Gómez-Huélamo, C., Gutiérrez, R., & Díaz-Díaz, A. (2022). Deep reinforcement learning based control for Autonomous Vehicles in CARLA. *Multimedia Tools and Applications*, 81(3), 3553-3576.

**CARLA Simulator:**
[carla.org](https://carla.org)

## 📝 Notas Importantes

1. **Guardado automático**: Checkpoints cada 50 episodios
2. **Renders desactivados**: Durante training para velocidad
3. **Evaluación sin exploración**: eps=0 para medir capacidad real
4. **Rutas aleatorias**: Generalización mejor que rutas fijas
5. **Paper usa DDPG**: Nosotros implementamos DQN (versión discreta)

## ✅ Checklist de Implementación

- [x] Estado 123-dim (121 imagen + φt + dt)
- [x] Red 2-FC layers (64, 32 neurons)
- [x] 27 acciones discretas
- [x] DQN con experience replay
- [x] Training loop con episodios
- [x] Checkpoints cada 50 episodios
- [x] Evaluación con RMSE
- [x] Función recompensa del paper
- [x] Sensores colisión + invasión carril
- [x] Rutas aleatorias A* planner
- [x] Visualización waypoints
- [x] Camera chase mode
- [x] Video recording

## 🚀 Próximos Pasos

1. Entrenar 500 episodios
2. Evaluar mejor checkpoint (20 iteraciones)
3. Si RMSE > 0.15m → más training
4. Comparar con paper (Tabla 3)
5. Considerar implementar DDPG (mejor rendimiento)

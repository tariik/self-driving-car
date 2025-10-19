# 🚀 Guía Rápida: Entrenamiento con TensorBoard

## ✅ TensorBoard ya está integrado!

El sistema de logging con TensorBoard ya está completamente integrado en `src/main.py`.

---

## 📊 Uso

### 1. Iniciar el Entrenamiento

```bash
# Activar entorno virtual
source env/bin/activate

# Iniciar entrenamiento (TensorBoard se activa automáticamente)
python src/main.py
```

El entrenamiento ahora registra automáticamente:
- ✅ Rewards por episodio y por step
- ✅ Estado del vehículo (velocidad, distancia, ángulo)
- ✅ Acciones (throttle, steering, brake)
- ✅ Epsilon (exploración)
- ✅ Colisiones e invasiones de carril
- ✅ Promedios móviles (10 y 100 episodios)

### 2. Visualizar con TensorBoard

**Opción A - Script automático:**
```bash
./start_tensorboard.sh
```

**Opción B - Comando manual:**
```bash
tensorboard --logdir=runs
```

Luego abre en tu navegador:
```
http://localhost:6006
```

---

## 📈 Métricas Principales

### Por Episodio (`Episode/*`)
- **Total_Reward**: Suma de todos los rewards del episodio
- **Length**: Número de steps completados
- **Avg_Reward_Per_Step**: Reward promedio = Total/Length
- **Collision**: Si terminó por colisión (0 o 1)
- **Lane_Invasion**: Si salió del carril (0 o 1)

### Estado del Vehículo (`State/*`)
Según el paper (dft = (vt, dt, φt)):
- **Velocity_vt**: Velocidad en m/s
- **Distance_dt**: Distancia al centro del carril en metros
- **Angle_phi_t**: Ángulo con respecto al carril en grados

### Acciones (`Action/*`)
- **Throttle**: Aceleración [0, 1]
- **Steering**: Dirección [-1, 1]
- **Brake**: Frenado [0, 1]

### Entrenamiento (`Training/*`)
- **Epsilon**: Tasa de exploración (DQN epsilon-greedy)
  - Inicia en 1.0 (100% exploración)
  - Decae exponencialmente
  - Termina en 0.01 (1% exploración)

### Acumuladas (`Cumulative/*`)
- **Collision_Rate**: Proporción de episodios con colisión
- **Lane_Invasion_Rate**: Proporción de episodios con invasión

### Promedios Móviles (`Running_Avg_10/*` y `Running_Avg_100/*`)
- **Reward**: Promedio móvil de rewards
- **Length**: Promedio móvil de longitudes

### Mejor Modelo (`Best/*`)
- **Episode**: Episodio del mejor modelo
- **Reward**: Mejor reward alcanzado

---

## 🎯 Cómo Interpretar las Gráficas

### 1. Total Reward (Principal Métrica)
```
📈 Buena señal: Tendencia ascendente
📉 Mala señal: Estancado o descendente
🎯 Objetivo: Maximizar reward (menos colisiones, mejor control)
```

### 2. Episode Length
```
📈 Buena señal: Episodios más largos (sobrevive más)
📉 Mala señal: Episodios muy cortos (colisiona rápido)
🎯 Objetivo: Completar rutas largas sin colisionar
```

### 3. Epsilon (Exploración)
```
📉 Esperado: Decaimiento exponencial de 1.0 → 0.01
🎯 Objetivo: Al inicio explora, al final explota conocimiento
```

### 4. Collision Rate
```
📉 Buena señal: Disminuye con el tiempo
📈 Mala señal: Se mantiene alta o aumenta
🎯 Objetivo: Tender a 0 (sin colisiones)
```

### 5. Estado del Vehículo
```
Velocity: Debería estabilizarse en velocidades seguras (5-15 m/s)
Distance: Cerca de 0 (centrado en el carril)
Angle: Cerca de 0° (alineado con el carril)
```

---

## 🔍 Comparar Múltiples Experimentos

TensorBoard permite comparar múltiples runs:

1. Cada entrenamiento crea un nuevo directorio en `runs/`
2. En TensorBoard, marca los checkboxes de los runs a comparar
3. Las gráficas se superponen para comparación directa

**Útil para:**
- Comparar hiperparámetros
- Ver efecto de epsilon decay
- Evaluar diferentes arquitecturas

---

## 💾 Estructura de Logs

```
runs/
├── DQN_Flatten_20251019_143025/
│   ├── events.out.tfevents.xxx
│   └── (logs de este experimento)
├── DQN_Flatten_20251019_150430/
│   └── (logs de otro experimento)
└── ...
```

Cada experimento tiene timestamp único.

---

## 🎓 Métricas del Paper

El paper Pérez-Gil et al. (2022) usa:

| Métrica del Paper | En TensorBoard |
|-------------------|----------------|
| Total reward | `Episode/Total_Reward` |
| Episode length | `Episode/Length` |
| Velocity (vt) | `State/Velocity_vt` |
| Distance (dt) | `State/Distance_dt` |
| Angle (φt) | `State/Angle_phi_t` |
| Collision events | `Episode/Collision` |
| Training episodes | HParams |

Todas las métricas clave están implementadas! ✅

---

## 🚨 Troubleshooting

### TensorBoard no inicia
```bash
# Verificar instalación
pip install tensorboard

# Verificar que existe el directorio
ls -la runs/

# Iniciar con verbose
tensorboard --logdir=runs --verbose
```

### No se ven gráficas
- Espera a que termine al menos 1 episodio
- Refresca el navegador (F5)
- Verifica que el entrenamiento está corriendo

### Puerto 6006 ocupado
```bash
# Usar otro puerto
tensorboard --logdir=runs --port=6007
```

---

## 📚 Recursos

- **Paper**: Pérez-Gil et al. (2022) - DOI: 10.1007/s11042-021-11437-3
- **TensorBoard Docs**: https://www.tensorflow.org/tensorboard
- **Guía completa**: Ver `TENSORBOARD_GUIDE.md`

---

## ✨ Resumen

```bash
# 1. Iniciar entrenamiento
python src/main.py

# 2. En otra terminal, iniciar TensorBoard
./start_tensorboard.sh

# 3. Abrir navegador
http://localhost:6006

# 4. ¡Disfrutar visualizando el progreso en tiempo real! 📊
```

**¡TensorBoard ya está completamente integrado y funcionando!** 🎉

# 🎯 Guía de Evaluación del Agente DRL-Flatten-Image

## 📚 Basado en el Paper
**Pérez-Gil et al. 2022** - Sección 6.1.2 (Validation Stage)

El paper evalúa cada agente conduciéndolo 20 veces por la misma ruta y calculando:
- **RMSE** (Root Mean Square Error): Error promedio respecto a trayectoria ideal
- **Error Máximo**: Mayor desviación en toda la ruta  
- **Tiempo Promedio**: Tiempo en completar la ruta

## 🚀 Uso Rápido

### 1. Evaluar el último checkpoint entrenado
```bash
python evaluate_agent.py
```

### 2. Evaluar un checkpoint específico
```bash
python evaluate_agent.py --checkpoint checkpoints/drl_flatten_episode_200.pth
```

### 3. Evaluar con más iteraciones (mayor precisión)
```bash
python evaluate_agent.py --iterations 50
```

### 4. Evaluar sin video (más rápido)
```bash
python evaluate_agent.py --no-video
```

### 5. Evaluar sin display (headless)
```bash
python evaluate_agent.py --no-display
```

## 📊 Métricas Calculadas

### Precisión
- **RMSE promedio**: Distancia promedio de la trayectoria real vs ideal
- **Error máximo**: Mayor desviación encontrada
- **Desviación estándar**: Consistencia del agente

### Rendimiento
- **Tasa de éxito**: % de rutas completadas sin colisión/invasión
- **Tiempo promedio**: Eficiencia temporal
- **Recompensa total**: Métrica de aprendizaje

### Comparación con Paper (Tabla 3)
```
Método                    RMSE (m)    Max Error (m)
----------------------------------------
LQR (baseline)            0.06        0.74
DQN-Flatten-Image         0.08        -
DDPG-Flatten-Image        0.07        -
Tu modelo                 ???         ???
```

## 🎯 Criterios de Calidad

| RMSE       | Evaluación                |
|------------|---------------------------|
| < 0.10 m   | 🏆 Excelente              |
| < 0.15 m   | ✅ Bueno                  |
| < 0.20 m   | ⚠️  Aceptable             |
| > 0.20 m   | ❌ Necesita más training  |

## 📁 Outputs Generados

### 1. Video de evaluación
- **Ubicación**: `evaluation_output/`
- **Contenido**: Grabación de las iteraciones con HUD
- **Formato**: MP4 (creado con ffmpeg)

### 2. Archivo de resultados
- **Nombre**: `evaluation_results_episode_XXX.txt`
- **Contenido**: Métricas detalladas y comparación con paper

### 3. Console output
- Resumen por iteración
- Estadísticas finales
- Comparación con benchmarks

## 🔧 Opciones Avanzadas

```bash
# Evaluación completa con display y video
python evaluate_agent.py \
    --checkpoint checkpoints/drl_flatten_final.pth \
    --iterations 20

# Evaluación rápida sin video
python evaluate_agent.py \
    --checkpoint checkpoints/drl_flatten_episode_100.pth \
    --iterations 5 \
    --no-video

# Evaluación headless (servidor sin X)
python evaluate_agent.py \
    --no-display \
    --no-video
```

## 📈 Interpretación de Resultados

### ✅ Agente Bien Entrenado
```
RMSE promedio: 0.075 ± 0.012 m
Error máximo promedio: 0.423 m
Tasa de éxito: 95.0% (19/20)
Colisiones: 0
Invasiones de carril: 1
```

### ⚠️ Agente Necesita Más Training
```
RMSE promedio: 0.234 ± 0.089 m
Error máximo promedio: 1.567 m
Tasa de éxito: 65.0% (13/20)
Colisiones: 4
Invasiones de carril: 3
```

## 🐛 Troubleshooting

### Error: "No se encontraron checkpoints"
```bash
# Primero entrena un modelo
python src/main.py
```

### Error: "CARLA no está corriendo"
```bash
# En otra terminal, lanza CARLA primero
./launch_carla.sh
```

### Display no funciona (headless)
```bash
# Usa modo headless
python evaluate_agent.py --no-display
```

## 📝 Notas Importantes

1. **Modo de Evaluación**: El agente NO explora (eps=0), solo explota lo aprendido
2. **Rutas Aleatorias**: Cada iteración usa una ruta diferente (como en training)
3. **Sensores**: Usa misma configuración que training (11×11, φt, dt)
4. **Determinismo**: Con el mismo checkpoint, resultados pueden variar por rutas aleatorias

## 🎓 Siguientes Pasos

1. **Si RMSE > 0.15m**: Continuar training (más episodios)
2. **Si tasa éxito < 80%**: Revisar función de recompensa
3. **Si muchas colisiones**: Aumentar penalización por colisión
4. **Si buen RMSE pero lento**: Entrenar con recompensa de velocidad

## 📚 Referencias

- Paper: Pérez-Gil, Ó., Barea, R., López-Guillén, E. et al. (2022). 
  "Deep reinforcement learning based control for Autonomous Vehicles in CARLA"
  *Multimedia Tools and Applications*, 81, 3553–3576

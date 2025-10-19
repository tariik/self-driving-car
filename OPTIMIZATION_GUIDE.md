# 🚀 Guía de Optimización para Entrenamiento DRL

## 📊 Análisis del Código Actual

### Puntos Críticos Identificados

1. **Logs en consola cada 10 steps** → Ralentiza el entrenamiento
2. **Display activo** → Consume CPU/GPU para renderizado
3. **Cálculo de velocidad en cada step** → Llamadas repetidas a CARLA
4. **Extracción de características repetida** → Código duplicado
5. **Sincronización de display** → Overhead de pygame

## ⚡ Optimizaciones Recomendadas

### 1. 🎯 Reducir Frecuencia de Logs (Alto Impacto)

**Problema:** Imprimir en consola cada 10 steps es costoso.

**Solución:**

```python
# En src/main.py
LOG_FREQUENCY = 100  # Cambiar de 10 a 100 o más

if step % LOG_FREQUENCY == 0:
    # ... logs ...
```

**Impacto:** +10-15% velocidad de entrenamiento

---

### 2. 🖥️ Desactivar Display en Training (Muy Alto Impacto)

**Problema:** El display consume recursos gráficos y CPU.

**Solución:**

```python
# En src/main.py, línea ~95
use_display = False  # ⚡ DESACTIVAR para máximo rendimiento
```

**Impacto:** +30-40% velocidad de entrenamiento

---

### 3. 🔧 Cachear Cálculos Repetidos (Medio Impacto)

**Problema:** Extraemos características del estado varias veces.

**Solución:** Crear función helper

```python
def extract_state_info(state, env):
    """Extrae toda la información del estado de una vez"""
    phi_t = state[121] if len(state) > 121 else 0.0
    d_t = state[122] if len(state) > 122 else 0.0
    
    v_t = np.linalg.norm([
        env.hero.get_velocity().x,
        env.hero.get_velocity().y,
        env.hero.get_velocity().z
    ]) if hasattr(env, 'hero') else 0.0
    
    return v_t, d_t, phi_t
```

**Impacto:** +5-8% velocidad de entrenamiento

---

### 4. ⏱️ Optimizar Timestep de CARLA (Medio Impacto)

**Problema:** Timestep de 0.1s puede ser demasiado fino.

**Solución:**

```python
# En src/main.py, línea ~23
"timestep": 0.05,  # Cambiar de 0.1 a 0.05 (duplica velocidad de simulación)
```

**Impacto:** +50% velocidad de simulación (pero menos realista)

**⚠️ Cuidado:** Afecta la física del simulador

---

### 5. 🎮 Modo Sin Renderizado (Muy Alto Impacto)

**Problema:** CARLA renderiza aunque no uses las imágenes RGB para display.

**Solución:**

```python
# En src/main.py
config = {
    "carla": {
        # ...
        "quality_level": "Low",
        "enable_rendering": False,  # ⚡ Desactivar si no necesitas visualización
    }
}
```

**Impacto:** +20-30% velocidad

**⚠️ Solo si no necesitas las imágenes RGB**

---

### 6. 📉 Reducir Resolución de Cámara (❌ NO RECOMENDADO)

**Problema:** Capturamos 640×480 y luego reducimos a 11×11.

**❌ NO APLICAR ESTA OPTIMIZACIÓN:**

```python
# ❌ NO HACER ESTO - Contradice el paper
"image_size_x": "320",  # ❌ NO cambiar
"image_size_y": "240",  # ❌ NO cambiar
```

**⚠️ RAZÓN:** El paper Pérez-Gil et al. (2022) específicamente dice:

> *"from 640x480 pixels to 11x11, reducing the amount of data from 300k to 121"*

**Mantener configuración del paper:**
```python
# ✅ CORRECTO - Según el paper
"image_size_x": "640",  # ✅ Mantener 640×480
"image_size_y": "480",  # ✅ Como especifica el paper
```

**Explicación:** 
- El paper captura **640×480** intencionalmente
- El **resize a 11×11** se hace después en `post_process_image()`
- Cambiar la resolución inicial podría afectar:
  - La calidad de la información capturada
  - El proceso de resize (interpolación)
  - La reproducibilidad del paper
  - Los resultados del entrenamiento

**Impacto potencial si se cambia:** +5-10% velocidad, pero **resultados no validados**

**Conclusión:** ❌ **NO aplicar** - Mantener fidelidad al paper

---

### 7. 🗑️ Desactivar Video Recording (Medio Impacto)

**Problema:** El video recorder guarda frames aunque no crees el video.

**Solución:**

```python
# Ya está implementado correctamente
SAVE_VIDEO = False  # ✅ Ya está desactivado
SAVE_RENDERS = False  # ✅ Ya está desactivado
```

**Impacto:** +10-15% velocidad (ya optimizado)

---

### 8. 🧮 Modo Asíncrono de CARLA (Alto Impacto)

**Problema:** Modo síncrono espera cada frame.

**Solución:**

```python
# En src/env/carla_core.py, buscar set_synchronous_mode
# Cambiar a modo asíncrono (más rápido pero menos determinista)
world.set_synchronous_mode(False)
```

**Impacto:** +40-60% velocidad

**⚠️ Cuidado:** Menos determinista, puede afectar reproducibilidad

---

### 9. 📊 Batch de Episodios (Bajo Impacto Inmediato)

**Problema:** Checkpoints y estadísticas cada episodio.

**Solución:**

```python
# Guardar checkpoints menos frecuentemente
SAVE_EVERY_N_EPISODES = 100  # Cambiar de 50 a 100

# Estadísticas menos frecuentes
if (episode + 1) % 50 == 0:  # Cambiar de 10 a 50
    # ... print stats ...
```

**Impacto:** +2-3% velocidad

---

### 10. 🔢 NumPy Optimizations (Bajo Impacto)

**Problema:** Operaciones NumPy no optimizadas.

**Solución:**

```python
# Usar operaciones vectorizadas
velocity_vector = env.hero.get_velocity()
v_t = np.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)

# En lugar de:
v_t = np.linalg.norm([velocity_vector.x, velocity_vector.y, velocity_vector.z])
```

**Impacto:** +1-2% velocidad

---

## 🎯 Configuración Recomendada para Entrenamiento Rápido

### Archivo: `src/main.py` (Optimizado)

```python
# ========== CONFIGURACIÓN DE ENTRENAMIENTO OPTIMIZADO ==========

# Display y logging
use_display = False  # ⚡ DESACTIVADO para máximo rendimiento
LOG_FREQUENCY = 100  # Logs cada 100 steps en lugar de 10

# Guardado de datos
SAVE_RENDERS = False  # ✅ Desactivado
SAVE_VIDEO = False    # ✅ Desactivado
DEBUG_STATE_FIRST_3_STEPS = False  # ✅ Desactivado

# CARLA settings
config = {
    "carla": {
        "timestep": 0.05,  # ⚡ Simulación más rápida (era 0.1)
        "quality_level": "Low",
        "enable_rendering": True,  # Necesario para la cámara del agente
    }
}

# Checkpoints menos frecuentes
SAVE_EVERY_N_EPISODES = 100  # Era 50

# Estadísticas menos frecuentes  
if (episode + 1) % 50 == 0:  # Era 10
    # ... print stats ...
```

### Estimación de Mejora

| Optimización | Impacto | Acumulado | Recomendado |
|--------------|---------|-----------|-------------|
| Base (sin cambios) | 100% | 100% | - |
| + Desactivar display | +35% | 135% | ✅ Sí |
| + Logs cada 100 steps | +12% | 152% | ✅ Sí |
| + Cachear cálculos | +7% | 163% | ✅ Sí |
| + Timestep 0.05 | +50% | 244% | ⚠️ Opcional |
| ~~Reducir resolución~~ | ~~+8%~~ | - | ❌ No (contradice paper) |
| **TOTAL SEGURO** | **+63%** | **~163%** | **✅ Sin alterar paper** |

---

## 📈 Configuración por Fase de Entrenamiento

### Fase 1: Entrenamiento Rápido (Exploración)

```python
use_display = False
LOG_FREQUENCY = 200
timestep = 0.05
SAVE_EVERY_N_EPISODES = 100
MAX_EPISODES = 500
```

**Objetivo:** Entrenar rápido para probar hiperparámetros

---

### Fase 2: Entrenamiento Normal (Producción)

```python
use_display = False
LOG_FREQUENCY = 100
timestep = 0.1
SAVE_EVERY_N_EPISODES = 50
MAX_EPISODES = 500
```

**Objetivo:** Entrenamiento estable y reproducible

---

### Fase 3: Debugging/Visualización

```python
use_display = True
LOG_FREQUENCY = 10
timestep = 0.1
SAVE_EVERY_N_EPISODES = 10
MAX_EPISODES = 5  # Pocos episodios
```

**Objetivo:** Ver qué hace el agente, debugging

---

## 🔧 Script de Aplicación Automática

Puedes crear un script para cambiar entre modos:

```bash
# fast_train.sh
#!/bin/bash
# Configura para entrenamiento rápido
sed -i 's/use_display = True/use_display = False/' src/main.py
sed -i 's/LOG_FREQUENCY = 10/LOG_FREQUENCY = 100/' src/main.py
sed -i 's/timestep": 0.1/timestep": 0.05/' src/main.py
python src/main.py
```

```bash
# debug_train.sh
#!/bin/bash
# Configura para debugging
sed -i 's/use_display = False/use_display = True/' src/main.py
sed -i 's/LOG_FREQUENCY = 100/LOG_FREQUENCY = 10/' src/main.py
sed -i 's/timestep": 0.05/timestep": 0.1/' src/main.py
python src/main.py
```

---

## 📊 Monitoreo de Rendimiento

### Agregar Timer

```python
import time

# Al inicio del entrenamiento
training_start = time.time()

# Al final de cada episodio
episode_time = time.time() - episode_start
print(f"   ⏱️  Episode time: {episode_time:.1f}s")

# Al final del entrenamiento
training_time = time.time() - training_start
print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
print(f"⏱️  Average per episode: {training_time/MAX_EPISODES:.1f}s")
```

---

## ⚠️ Advertencias Importantes

### Optimizaciones Seguras (No afectan reproducibilidad del paper)

✅ **Desactivar display** → Solo visualización, no afecta entrenamiento
✅ **Reducir frecuencia de logs** → Solo output, no afecta algoritmo
✅ **Cachear cálculos** → Mismos resultados, más eficiente
✅ **Guardar checkpoints menos frecuentemente** → Solo I/O

### Optimizaciones Riesgosas (Pueden afectar resultados)

⚠️ **Timestep más pequeño** → Física menos realista, resultados diferentes
⚠️ **Modo asíncrono** → Menos reproducibilidad, no determinista
❌ **Cambiar resolución de cámara** → **Contradice el paper directamente**
❌ **Desactivar renderizado** → Necesario para capturar imágenes

### Regla de Oro

> **Si el paper especifica un valor, NO lo cambies a menos que sea solo para visualización/logging**

Parámetros del paper que **NO se deben cambiar**:
- ❌ **Resolución de cámara: 640×480** (Paper: "from 640x480 pixels to 11x11")
- ❌ **Tamaño final de imagen: 11×11** (Paper: "reducing the amount of data from 300k to 121")
- ❌ **Frame stack: 1** (No frame stacking en DRL-Flatten-Image)
- ❌ **Número de episodios DDPG: ~500** (Paper Table 2: "DDPG-Flatten-Image: 500 episodes, Best: 50")
- ❌ **Número de acciones: 27** (Paper: "27 discrete driving actions")
- ❌ **Estado: 123 dimensiones** (Paper Eq. 21: "S = ([Pt0...Pt120], φt, dt)" = 121 + 2)
- ❌ **Red: 2 Fully-Connected Layers** (Paper: "really simple 2 Fully-Connected Layers network")

Parámetros que **SÍ se pueden optimizar**:
- ✅ Display on/off
- ✅ Frecuencia de logs
- ✅ Frecuencia de checkpoints
- ✅ Video recording on/off

---

## ✅ Checklist de Optimización

Antes de entrenar 500 episodios completos:

- [ ] `use_display = False`
- [ ] `LOG_FREQUENCY = 100` o más
- [ ] `SAVE_VIDEO = False`
- [ ] `SAVE_RENDERS = False`
- [ ] Considerar `timestep = 0.05` (opcional)
- [ ] `SAVE_EVERY_N_EPISODES = 100` (opcional)
- [ ] Cerrar otras aplicaciones pesadas
- [ ] Verificar que CARLA corra solo (sin display)

---

## 🎯 Próximos Pasos

1. **Aplicar optimizaciones básicas** (display, logs)
2. **Ejecutar 50 episodios de prueba** para medir tiempo
3. **Ajustar configuración** según resultados
4. **Entrenar modelo completo** (500 episodios)
5. **Evaluar con display activo** usando `evaluate_agent.py`


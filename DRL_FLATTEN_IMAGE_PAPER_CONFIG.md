# 📋 Configuración EXACTA: DRL-Flatten-Image del Paper

**Paper**: Pérez-Gil et al. (2022) - "Deep reinforcement learning based control for Autonomous Vehicles in CARLA"

---

## 🎯 CONFIGURACIÓN EXACTA DEL PAPER

### **Estado (State) - Ecuación (21)**

```
S = ([Pt0, Pt1, Pt2, ...Pt120], φt, dt)
```

**Componentes:**
1. **Imagen B/W**: 640x480 → redimensionada a **11x11** → aplanada a **121 valores**
2. **φt**: Ángulo al carril (rad)
3. **dt**: Distancia al centro del carril (m)

**Total**: 121 + 1 + 1 = **123 dimensiones**

### **Acción (Action)**

- **DQN**: 27 acciones discretas (Tabla 1)
  - Steering: -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1 (9 valores)
  - Throttle: 0, 0.5, 1 (3 valores)
  - Total: 9 × 3 = 27 combinaciones

- **DDPG**: Acciones continuas
  - Steering: [-1, 1]
  - Throttle: [0, 1]

### **Reward Function - Ecuaciones (18-19)**

```python
# Ecuación (18)
R = -200  if collision or lane_change or roadway_departure

# Ecuación (19)
R = Σ |vt·cos(φt)| - |vt·sin(φt)| - |vt|·|dt|  if car in lane
```

### **Red Neuronal**

**Arquitectura (del paper - línea 560):**
- Input: 123 dimensiones (121 imagen + φt + dt)
- 2 Fully-Connected Layers (tamaño no especificado exactamente)
- Output: 27 acciones (DQN) o 2 acciones (DDPG)

### **Entrenamiento**

| Parámetro | DQN-Flatten-Image | DDPG-Flatten-Image |
|-----------|-------------------|---------------------|
| **Total Episodes** | 20,000 | 500 |
| **Best Episode** | 16,500 | 50 |
| **Mapa** | Town01 | Town01 |
| **Rutas** | Aleatorias (training) | Aleatorias (training) |
| **Ruta validación** | Fija ~180m | Fija ~180m |

### **Condiciones de Término de Episodio**

Según líneas 714-717 del paper:

```python
done = True if:
    - collision_sensor activated
    - lane_invasor activated
    - max episodes reached
```

### **Hardware (línea 726-727)**

```
CPU: Intel Core i7-9700k
RAM: 32GB
GPU: NVIDIA GeForce RTX 2080 Ti (11GB VRAM)
```

---

## ⚠️ DIFERENCIAS CON TU CÓDIGO ACTUAL

| Aspecto | Paper | Tu Código Actual | Estado |
|---------|-------|------------------|--------|
| **Tamaño imagen** | 11x11 (121 valores) | 84x84 (7,056 valores/frame) | ❌ DIFERENTE |
| **Frame stack** | 1 frame | 4 frames | ❌ DIFERENTE |
| **Estado total** | 123 dim (121 + φt + dt) | 28,224 dim (84×84×4) | ❌ DIFERENTE |
| **φt, dt en estado** | ✅ SÍ incluidos | ❌ NO incluidos | ❌ DIFERENTE |
| **Reward -200** | ✅ | ✅ | ✅ CORRECTO |
| **Episode ends** | ✅ collision + lane | ✅ | ✅ CORRECTO |
| **Rutas aleatorias** | ✅ | ✅ | ✅ CORRECTO |
| **Mapa Town01** | ✅ | ✅ | ✅ CORRECTO |
| **27 acciones** | ✅ | ✅ | ✅ CORRECTO |

---

## 🔧 CAMBIOS NECESARIOS

### 1. **Cambiar tamaño de imagen: 84x84 → 11x11**

```python
# En config de cámara
"image_size_x": "11",  # Cambiar de 84
"image_size_y": "11",  # Cambiar de 84
"size": 11  # Para observation space
```

### 2. **Eliminar frame stacking (usar solo 1 frame)**

```python
# En base_env.py
"framestack": 1,  # Cambiar de 4 a 1
```

### 3. **Agregar φt y dt al estado**

```python
# En base_env.py - get_observation()
# Concatenar: [imagen_11x11_flatten] + [φt] + [dt]
driving_features = self.get_driving_features(core)
image_flat = image.flatten()  # 121 valores
state = np.concatenate([image_flat, [driving_features[2]], [driving_features[1]]])
# Total: 121 + 1 + 1 = 123 dimensiones
```

### 4. **Actualizar red neuronal**

```python
# En drl_flatten_agent.py
class QNetwork(nn.Module):
    def __init__(self, state_size=123, action_size=27, seed=0):
        # state_size = 123 (121 imagen + 1 φt + 1 dt)
        # fc1: 123 → 64
        # fc2: 64 → 32
        # fc3: 32 → 27
```

---

## 📊 RESULTADOS ESPERADOS (según paper)

### **DQN-Flatten-Image:**
- Training episodes: 20,000
- Best episode: 16,500
- RMSE: ~0.0968m (Tabla 3, línea 854)
- Tiempo promedio: ~11.8s en ruta de 180m

### **DDPG-Flatten-Image:**
- Training episodes: 500
- Best episode: 50
- RMSE: ~0.0956m (Tabla 3, línea 858)
- Tiempo promedio: ~11.0s en ruta de 180m
- **50x más rápido que DQN** (50 vs 16,500 episodios)

---

## ✅ PLAN DE ACCIÓN

1. ✅ Cambiar tamaño imagen a 11x11
2. ✅ Cambiar framestack a 1
3. ✅ Agregar φt y dt al estado
4. ✅ Actualizar observation_space a 123
5. ✅ Actualizar red neuronal (input 123)
6. ✅ Verificar 27 acciones discretas
7. ✅ Confirmar Town01
8. ✅ Confirmar rutas aleatorias
9. ✅ Entrenar y comparar con resultados del paper

¿Quieres que implemente estos cambios ahora? 🚀

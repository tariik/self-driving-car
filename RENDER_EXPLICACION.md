# RENDER: VISUALIZACIÓN EXACTA DE LO QUE VE EL AGENTE

## 🎯 Cambio Realizado

**ANTES**: Render mostraba imagen de 336×336 (visualización arbitraria)

**AHORA**: Render muestra **EXACTAMENTE** la imagen de 11×11 que recibe el agente

---

## 📸 Detalles Técnicos

### Proceso de Render (src/env/carla_env.py):

```python
def render(self, mode='human'):
    """
    Guarda la observación exacta del agente como imagen.
    
    Estado del agente: [121 píxeles imagen, φt, dt]
    Total: 123 valores
    """
    
    # 1. Extraer imagen del estado (primeros 121 valores)
    image_flat = observation[:121]
    
    # 2. Reshape a 11×11
    image_2d = image_flat.reshape(11, 11)
    
    # 3. Desnormalizar de [-1, 1] a [0, 255]
    image_2d = ((image_2d * 128) + 128).clip(0, 255).astype(np.uint8)
    
    # 4. Escalar a 330×330 (11×30) para visualización
    #    Usa NEAREST para mantener píxeles cuadrados
    img = img.resize((330, 330), Image.NEAREST)
    
    # 5. Agregar texto con φt y dt
    # 6. Guardar como frame_XXXX.png
```

---

## 📊 Características de las Imágenes Guardadas

| Propiedad | Valor |
|-----------|-------|
| **Tamaño guardado** | 330×330 píxeles |
| **Tamaño real del agente** | 11×11 píxeles |
| **Escala** | 30x (330÷11 = 30) |
| **Interpolación** | NEAREST (píxeles cuadrados) |
| **Formato** | PNG RGB con texto |
| **Información adicional** | φt (ángulo) y dt (distancia) |

---

## 🖼️ Qué Verás en las Imágenes

### ✅ NORMAL (esperado):
- **MUY pixelada** (solo 11×11 = 121 cuadrados)
- Cada "píxel" es un cuadrado de 30×30 en la imagen guardada
- Difícil de reconocer detalles
- Grayscale (blanco y negro)
- Texto verde mostrando:
  - `Agente ve: 11x11 px`
  - `φt: X.XXXX rad (X.XX°)` - Ángulo al carril
  - `dt: X.XXXX m` - Distancia al centro

### ❌ PROBLEMA (no esperado):
- Imagen completamente negra o blanca
- Imagen con colores (debería ser B/W)
- Imagen no pixelada (píxeles suaves)

---

## 📁 Ubicación

```bash
render_output/
├── frame_0000.png  # Primera observación
├── frame_0001.png  # Después de step 1
├── frame_0002.png  # Después de step 2
└── ...
```

---

## 🔍 Ejemplo de Visualización

```
Original CARLA: 640×480 RGB
       ↓
Procesada: 11×11 B/W (lo que ve el agente)
       ↓
Guardada: 330×330 RGB (escalada para visualización)
```

**Cada cuadrado de 30×30 píxeles en la imagen guardada = 1 píxel que ve el agente**

---

## 💡 Por Qué es Tan Pixelada

Del paper (Pérez-Gil et al. 2022):

> "This proposed agent reshapes the B/W frontal image, taken from the vehicle,
> from 640x480 pixels to 11x11, reducing the amount of data from 300k to 121."

**Ventajas de 11×11:**
1. ✅ Espacio de estados muy reducido (121 vs 307,200)
2. ✅ Entrenamiento más rápido
3. ✅ Red neuronal más simple
4. ✅ Funciona bien según resultados del paper

**La imagen pixelada NO es un bug, es el DISEÑO del paper!**

---

## 🚀 Uso

### En entrenamiento:
```python
env = CarlaEnv(experiment, config)
state = env.reset()

for step in range(max_steps):
    action = agent.select_action(state)
    next_state, reward, done, _, info = env.step(action)
    env.render()  # Guarda frame_XXXX.png
    
    if done:
        break
```

### Ver resultados:
```bash
# Las imágenes se guardan automáticamente en:
ls render_output/

# Ver con cualquier visor de imágenes
eog render_output/frame_0000.png
```

---

## 🎯 Comparación: Antes vs Ahora

| Aspecto | ANTES | AHORA |
|---------|-------|-------|
| **Origen** | Sensor raw 336×336 | Estado del agente 11×11 |
| **Representa** | Vista arbitraria | **EXACTAMENTE** lo que ve el agente |
| **Utilidad** | Debug genérico | Verificar input del agente |
| **Precisión** | Aproximada | **100% exacta** |

---

## ✅ Verificación

Para verificar que las imágenes son correctas:

```bash
python test_render.py
```

Esto creará 5 frames en `render_output/` mostrando exactamente lo que el agente ve en cada step.

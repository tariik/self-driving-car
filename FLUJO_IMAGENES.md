# FLUJO DE IMÁGENES EN EL PROYECTO

## 📸 Dos flujos de imagen separados:

### 1️⃣ IMAGEN DEL AGENTE (DRL Input)
```
CARLA Sensor → 640×480 RGB → Grayscale → 11×11 → Normalizar → Flatten → [121 valores]
                                                                              ↓
                                                                    Concatenar con φt, dt
                                                                              ↓
                                                                        [123 valores]
                                                                              ↓
                                                                         Red Neuronal
```

**Características:**
- Captura: 640×480 RGB desde sensor frontal
- Procesa: Convierte a B/W y reduce a 11×11
- Propósito: Input del agente DRL (muy eficiente)
- Archivo: `src/env/base_env.py` → `post_process_image()`

### 2️⃣ IMAGEN DE VISUALIZACIÓN (Render Output)
```
CARLA Last Observation → 336×336 RGB → Guardar PNG
```

**Características:**
- Tamaño: 336×336 píxeles (para que humanos puedan ver)
- Propósito: Debug y visualización
- NO es lo que el agente ve
- Carpeta: `render_output/frame_XXXX.png`
- Archivo: `src/env/carla_env.py` → `render()`

---

## 🔍 CONFIGURACIÓN ACTUAL (Paper Pérez-Gil et al. 2022)

### Sensor de Cámara RGB:
```python
"rgb_camera": {
    "type": "sensor.camera.rgb",
    "transform": "1.5,0.0,1.5,0.0,0.0,0.0",  # Frontal, parabrisas
    "image_size_x": "640",  # ✅ Paper: Original 640×480
    "image_size_y": "480",  # ✅ Paper: Original 640×480
    "fov": "90",            # Campo de visión 90°
    "size": 11              # ✅ Paper: Target resize 11×11
}
```

### Procesamiento (post_process_image):
1. **Captura**: CARLA devuelve 640×480 RGB
2. **Grayscale**: RGB → B/W (cv2.COLOR_RGB2GRAY)
3. **Resize**: 640×480 → 11×11 (cv2.resize)
4. **Normalizar**: [0,255] → [-1,1]
5. **Flatten**: (11,11,1) → (121,)
6. **Concatenar**: [121 píxeles] + [φt] + [dt] = [123 valores]

### Reducción de datos:
```
640 × 480 = 307,200 píxeles
11 × 11 = 121 píxeles
Reducción: 2,539× más pequeño
```

---

## ⚠️ IMPORTANTE: ¿Por qué 11×11 se ve "raro"?

**ES INTENCIONAL según el paper:**

> "This proposed agent reshapes the B/W frontal image, taken from the vehicle,
> from 640x480 pixels to 11x11, reducing the amount of data from 300k to 121."
> — Pérez-Gil et al. (2022), Sección 5.1

**Ventajas:**
1. ✅ Reducción dramática del espacio de estados
2. ✅ Entrenamiento más rápido (menos parámetros)
3. ✅ Suficiente información para navegación básica
4. ✅ Funciona bien según resultados del paper

**"Desventajas" (en realidad no lo son):**
- 🔲 Imagen MUY pixelada (solo 11×11 = 121 cuadrados)
- 🔲 Difícil de reconocer para humanos
- 🔲 Pérdida de detalles finos

**Pero el agente DRL no necesita ver como humanos!**

---

## 🎯 RESUMEN

Si ves la imagen del agente (11×11) y te parece "rara", **eso es CORRECTO**.
Si ves las imágenes de render_output (336×336) y se ven raras, **eso es diferente**.

¿Cuál de las dos es la que te parece rara?

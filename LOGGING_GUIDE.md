# 📊 Guía de Logging del Estado del Entorno

## Descripción

El sistema ahora incluye **logs profesionales** que muestran el estado del entorno durante el entrenamiento, tanto en la **consola (CMD)** como en la **ventana de visualización**.

## 📍 Información Mostrada

### Estado del Entorno (según Paper Pérez-Gil et al. 2022)

El estado `s_t` incluye:

- **v_t**: Velocidad del vehículo (m/s)
- **d_t**: Distancia al centro del carril (m)
- **φ_t**: Ángulo respecto al carril (radianes y grados)

### Acción Ejecutada

La acción `a_t` incluye:

- **Throttle**: Aceleración [0.0 - 1.0]
- **Steer**: Dirección [-1.0 a +1.0]
- **Brake**: Freno [0.0 - 1.0]

### Recompensa

- **r_t**: Recompensa del paso actual
- **R_total**: Recompensa acumulada del episodio

## 🖥️ Logs en Consola

Los logs aparecen en la consola **cada 10 steps** con el siguiente formato profesional:

```
======================================================================
  AGENT STATUS - Step   10
======================================================================
  Vehicle State:
    • Velocity:     12.34 m/s
    • Distance:      0.125 m     (to lane center)
    • Angle:        +5.23°       (lane alignment)
  ───────────────────────────────────────────────────────────────
  Control Action:
    • Throttle:     0.750        (acceleration)
    • Steering:    -0.120        (direction)
    • Brake:        0.000        (braking)
  ───────────────────────────────────────────────────────────────
  Training Metrics:
    • Reward:      +0.5432
    • Total:       +12.45
======================================================================
```

### Interpretación Profesional

**Vehicle State (Estado del Vehículo):**
- **Velocity**: Velocidad actual del vehículo en metros por segundo
- **Distance**: Distancia lateral al centro del carril en metros (ideal: ~0.0m)
- **Angle**: Ángulo de desalineación con la dirección del carril en grados (ideal: ~0°)

**Control Action (Acción de Control):**
- **Throttle**: Aceleración aplicada [0.0 - 1.0]
- **Steering**: Dirección del volante (negativo = izquierda, positivo = derecha) [-1.0 a +1.0]
- **Brake**: Frenado aplicado [0.0 - 1.0]

**Training Metrics (Métricas de Entrenamiento):**
- **Reward**: Recompensa obtenida en este step
- **Total**: Suma acumulada de todas las recompensas del episodio

## 🖼️ Display en Ventana de Visualización

Cuando el display está activado (`use_display = True` en `src/main.py`), se muestra un **HUD compacto** (Heads-Up Display) en la esquina superior derecha con la siguiente información:

```
╔══════════════════════════╗
║    AGENT STATUS          ║
║ ──────────────────────── ║
║ Vehicle State            ║
║  Velocity:   12.3 m/s   ║
║  Distance:    0.12 m    ║
║  Angle:      +5.2°      ║
║                          ║
║ Control Action           ║
║  Throttle:   0.75       ║
║  Steering:  -0.12       ║
║  Brake:      0.00       ║
║                          ║
║ Training                 ║
║  Step:       150        ║
║  Reward:    +0.543      ║
║                          ║
║ ● RUNNING                ║
╚══════════════════════════╝
```

**Características del HUD Profesional:**
- 📍 **Posición**: Esquina superior derecha (no tapa la vista principal)
- 📏 **Tamaño**: 220x240 píxeles (compacto pero legible)
- 🔍 **Transparencia**: 200/255 (semi-transparente)
- 🎨 **Diseño**: Fondo azul oscuro profesional con borde azul claro
- 📊 **Formato**: Etiquetas descriptivas con valores alineados
- 🎯 **Secciones**: Vehicle State (verde-azul), Control Action (naranja), Training (verde)
- 💡 **Status**: Badge con indicador visual (● RUNNING / ● TERMINATED)

## 🎛️ Control del Display

### Durante Entrenamiento

En `src/main.py`, línea ~95:

```python
use_display = True   # ACTIVADO: muestra ventana con logs
use_display = False  # DESACTIVADO: solo logs en consola (mejor rendimiento)
```

### Durante Evaluación

En `evaluate_agent.py`:

```bash
# Con display (por defecto)
python evaluate_agent.py --checkpoint checkpoints/drl_flatten_final.pth

# Sin display (solo logs en consola)
python evaluate_agent.py --checkpoint checkpoints/drl_flatten_final.pth --no-display
```

## 📈 Frecuencia de Logs

- **Consola**: Cada 10 steps (configurable en `src/main.py` línea ~197)
- **Display**: Actualización continua en tiempo real (30 FPS)

## 🔧 Personalización

### Cambiar Frecuencia de Logs en Consola

En `src/main.py`, línea ~197:

```python
if step % 10 == 0:  # Cambiar 10 a otro número
    print(...)
```

### Modificar Información Mostrada

Editar `src/utils/display_manager.py`, método `render_hud()` para agregar/quitar información.

## 🎯 Ejemplo de Uso

```bash
# 1. Asegúrate de que CARLA esté corriendo
./launch_carla.sh

# 2. Inicia el entrenamiento con display activado
python src/main.py

# 3. Observa los logs en consola y en la ventana
# - Logs en consola cada 10 steps
# - HUD en tiempo real en la ventana

# 4. Presiona ESC o Q para salir
```

## 📝 Notas

- Los logs muestran el **estado antes** de ejecutar la acción y el **reward después**
- La información es consistente entre consola y display
- El formato es compatible con análisis posterior (parsing fácil)
- Los colores en la ventana facilitan la interpretación:
  - **Cian**: Estado del entorno
  - **Magenta**: Acciones
  - **Verde**: Métricas de entrenamiento
  - **Amarillo**: Títulos/alertas

## 🐛 Troubleshooting

**Problema**: No veo logs en consola
- **Solución**: Verifica que el step % 10 == 0 esté descomentado

**Problema**: Display no muestra HUD
- **Solución**: Verifica que `use_display = True` en main.py

**Problema**: Los valores aparecen como 0.0
- **Solución**: Espera unos steps, el estado se inicializa gradualmente


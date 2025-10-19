# 👁️ Visualización de Waypoints en 3D

**Fecha**: 19 de octubre de 2025

---

## 🎯 Objetivo

Visualizar los waypoints de la ruta en el **mundo 3D de CARLA** para debugging y monitoreo, **SIN que aparezcan en la cámara del agente**.

---

## ✨ Características

### 🎨 Sistema de Colores

| Color | Significado | Descripción |
|-------|-------------|-------------|
| 🔵 **AZUL** | START | Punto de inicio de la ruta |
| 🟢 **VERDE** | Waypoints | Todos los puntos de la ruta |
| 🔴 **ROJO** | GOAL | Punto de destino final |
| 🟡 **AMARILLO** | Actual | Waypoint actual (se actualiza en cada step) |

### 📍 Elementos Visuales

1. **Puntos 3D** - Cada waypoint es un punto en el mundo
2. **Líneas** - Conectan waypoints consecutivos mostrando la ruta
3. **Texto** - Números cada 10 waypoints y labels START/GOAL
4. **Marcador dinámico** - Waypoint actual en amarillo (se actualiza)

### ⏱️ Persistencia

- **Lifetime**: 120 segundos (configurable)
- Los elementos permanecen visibles durante el training
- Se limpian automáticamente después del lifetime

---

## 🔧 Implementación

### Métodos Añadidos en `carla_env.py`

#### 1. `_visualize_route(waypoints, start_transform, end_transform)`

Visualiza toda la ruta cuando se genera:

```python
def _visualize_route(self, waypoints, start_transform, end_transform):
    """
    Visualiza la ruta en el mundo 3D de CARLA
    Los waypoints se dibujan SOLO en el spectator, NO en la cámara
    """
    debug = self.core.world.debug
    
    # Colores
    GREEN = carla.Color(0, 255, 0)
    BLUE = carla.Color(0, 100, 255)
    RED = carla.Color(255, 0, 0)
    
    # Dibujar START (azul)
    debug.draw_point(start_transform.location + carla.Location(z=0.5),
                     size=0.2, color=BLUE, life_time=120.0)
    
    # Dibujar GOAL (rojo)
    debug.draw_point(end_transform.location + carla.Location(z=0.5),
                     size=0.2, color=RED, life_time=120.0)
    
    # Dibujar waypoints (verde)
    for wp in waypoints:
        debug.draw_point(wp.transform.location + carla.Location(z=0.3),
                        size=0.05, color=GREEN, life_time=120.0)
    
    # Conectar con líneas
    for i in range(len(waypoints) - 1):
        debug.draw_line(waypoints[i].transform.location,
                       waypoints[i+1].transform.location,
                       color=GREEN, life_time=120.0)
```

#### 2. `_update_route_visualization()`

Actualiza el marcador del waypoint actual en cada step:

```python
def _update_route_visualization(self):
    """
    Actualiza la visualización para marcar el waypoint actual
    Llama esto en cada step para ver el progreso
    """
    idx = self.experiment.current_waypoint_index
    current_wp = self.experiment.route_waypoints[idx]
    
    # Marcador amarillo del waypoint actual
    self.core.world.debug.draw_point(
        current_wp.transform.location + carla.Location(z=0.5),
        size=0.15,
        color=carla.Color(255, 255, 0),  # Amarillo
        life_time=0.5
    )
```

### Integración en `step()`

```python
def step(self, action):
    # ...código existente...
    
    # Update route visualization (si hay ruta activa)
    if self.experiment.use_random_routes:
        self._update_route_visualization()
    
    return observation, reward, done, {}, info
```

---

## 🚀 Uso

### Activar Visualización

La visualización se activa automáticamente cuando `use_random_routes = True`:

```python
# Configuración
experiment_config = BASE_EXPERIMENT_CONFIG.copy()
experiment_config["use_random_routes"] = True  # ⭐ Activa visualización

# Crear entorno
experiment = BaseEnv(experiment_config)
env = CarlaEnv(experiment, config)

# Reset genera y visualiza ruta
obs, info = env.reset()
```

### Ver la Visualización

1. **En CARLA Server**: Los waypoints aparecen en el mundo 3D
2. **Spectator Camera**: Mueve la cámara para ver toda la ruta
3. **NO en RGB Camera**: La cámara del agente NO ve los waypoints

### Ejemplo Completo

```python
# Configurar
config["use_random_routes"] = True
env = CarlaEnv(experiment, config)

# Reset (genera y visualiza)
obs, info = env.reset()

# Training loop
for step in range(1000):
    action = agent.select_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    # 🟡 Waypoint actual se actualiza automáticamente
    
    if done:
        obs, info = env.reset()  # Nueva ruta visualizada
```

---

## 📊 Salida del Test

```
======================================================================
👁️  TEST: VISUALIZACIÓN DE WAYPOINTS EN 3D
======================================================================

🗺️  Ruta generada: 170 waypoints, 338.8m
👁️  Ruta visualizada: 170 waypoints (verde), START (azul), GOAL (rojo)

📊 Información de la ruta:
   • Total waypoints: 170
   • Distancia a meta: 192.8m

👁️  VISUALIZACIÓN ACTIVA:
   • Waypoints verdes: 170 puntos
   • Líneas verdes conectan los waypoints
   • Punto azul (START) al inicio
   • Punto rojo (GOAL) al final
   • Números cada 10 waypoints

🎮 SIGUIENDO LA RUTA (waypoint actual en AMARILLO)

   Step  5: Progreso   1.2% | WPs:   2/170 | 🟡 Waypoint actual marcado
   Step 10: Progreso   1.8% | WPs:   3/170 | 🟡 Waypoint actual marcado
   Step 30: Progreso   3.5% | WPs:   6/170 | 🟡 Waypoint actual marcado

✅ Visualización completada
```

---

## 💡 Casos de Uso

### 1. Debugging de Rutas

Ver si el A* planner genera rutas correctas:
```python
obs, info = env.reset()
# Inspeccionar visualmente en CARLA spectator
# ¿La ruta evita obstáculos?
# ¿Usa las calles correctas?
```

### 2. Monitoreo de Training

Ver si el agente sigue la ruta durante el entrenamiento:
```python
for episode in range(num_episodes):
    obs, info = env.reset()  # Nueva ruta visualizada
    
    while not done:
        action = agent.act(obs)
        obs, reward, done, _, info = env.step(action)
        # 🟡 Ver progreso en tiempo real
```

### 3. Análisis de Errores

Identificar dónde el agente se desvía:
```python
# Si el agente choca o sale del carril
# Ver en qué parte de la ruta ocurrió
# ¿Estaba cerca de un waypoint?
# ¿Qué tan desviado estaba?
```

### 4. Validación de Algoritmo

Verificar que el tracking de waypoints funciona:
```python
route_info = experiment.get_route_info(core)
print(f"Waypoint actual: {route_info['waypoints_completed']}")
# Ver 🟡 marcador amarillo en ese waypoint
```

---

## 🔍 API de CARLA Debug

### Métodos Usados

```python
debug = world.debug

# Dibujar punto
debug.draw_point(
    location,           # carla.Location
    size=0.1,          # Tamaño del punto
    color=carla.Color(r, g, b),  # Color RGB
    life_time=10.0     # Segundos (-1 = permanente)
)

# Dibujar línea
debug.draw_line(
    begin,             # carla.Location inicio
    end,               # carla.Location fin
    thickness=0.1,     # Grosor de la línea
    color=carla.Color(r, g, b),
    life_time=10.0
)

# Dibujar texto
debug.draw_string(
    location,          # carla.Location
    text="Texto",      # String a mostrar
    color=carla.Color(r, g, b),
    life_time=10.0
)
```

### Elevación (Z-offset)

Para que los elementos sean visibles sobre el suelo:
```python
# Punto a 0.3m sobre el suelo
location + carla.Location(z=0.3)

# Texto a 1.5m sobre el suelo (más visible)
location + carla.Location(z=1.5)
```

---

## ⚙️ Configuración Avanzada

### Cambiar Lifetime

```python
# En _visualize_route()
lifetime = 300.0  # 5 minutos

# Para permanente
lifetime = -1.0
```

### Cambiar Colores

```python
# Definir colores custom
CUSTOM_GREEN = carla.Color(100, 255, 100)  # Verde claro
CUSTOM_BLUE = carla.Color(50, 150, 255)    # Azul cielo
CUSTOM_RED = carla.Color(255, 100, 100)    # Rojo claro
```

### Cambiar Frecuencia de Números

```python
# Mostrar número cada 5 waypoints en vez de 10
if i % 5 == 0:
    debug.draw_string(...)
```

### Desactivar Visualización

```python
# Comentar la llamada en step()
# if self.experiment.use_random_routes:
#     self._update_route_visualization()

# O crear flag
self.visualize_route = False
if self.experiment.use_random_routes and self.visualize_route:
    self._update_route_visualization()
```

---

## ✅ Ventajas

1. **No afecta observaciones**: Los elementos debug NO aparecen en la cámara del agente
2. **Bajo overhead**: El rendering de debug es muy eficiente
3. **Tiempo real**: Se actualiza en cada step
4. **Intuitivo**: Colores y formas claras
5. **Configurable**: Lifetime, colores, tamaños ajustables

---

## ⚠️ Consideraciones

1. **Lifetime**: Elementos se acumulan si lifetime es muy largo
2. **Performance**: Muchos elementos pueden ralentizar el spectator (no el training)
3. **Visibilidad**: Solo en spectator, no en cámara del agente
4. **Servidor**: Requiere CARLA server con rendering habilitado

---

## 📝 Archivos

- **Implementación**: `src/env/carla_env.py`
  - `_visualize_route()` - Dibuja ruta completa
  - `_update_route_visualization()` - Actualiza waypoint actual
  
- **Test**: `test_waypoint_visualization.py`
  - Genera ruta y muestra visualización
  - Sigue ruta mostrando progreso
  - 30 steps con waypoint actual marcado

---

## 🎯 Próximos Pasos

Posibles mejoras:

1. **Flecha de dirección**: Mostrar orientación en cada waypoint
2. **Velocidad target**: Color según velocidad deseada
3. **Historial**: Traza del camino recorrido
4. **Colisiones**: Marcar puntos de colisión en rojo
5. **Invasiones**: Marcar puntos de invasión de carril

---

**Estado**: ✅ Implementado y testeado
**Archivo Test**: `test_waypoint_visualization.py`
**Integrado en**: Phase 1.4 - Random Routes

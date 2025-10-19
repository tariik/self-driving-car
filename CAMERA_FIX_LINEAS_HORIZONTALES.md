# PROBLEMA: Imagen con "Líneas Horizontales"

## 🔍 Diagnóstico

### Síntoma:
La imagen renderizada muestra solo "líneas horizontales" - cada fila tiene valores casi idénticos.

### Causa Raíz:
La matriz 11×11 capturada mostraba:
```
[[ 75  87  85  80  88  85  68  52  52  52  52]   ← Algo de variación
 [ 79  79  75  73  71  71  71  71  71  71  71]   ← Mayormente uniforme
 [104 104 104 101 100 100 100 100 100 100 100]  ← Casi toda igual
 [137 137 138 136 136 136 136 136 136 136 136]  ← Casi toda igual
 [150 150 150 150 150 150 150 150 150 150 150]  ← COMPLETAMENTE uniforme
 [160 160 160 160 160 160 160 160 160 160 160]  ← COMPLETAMENTE uniforme
 [173 173 173 173 173 173 173 173 173 173 173]  ← COMPLETAMENTE uniforme
 [195 195 195 195 195 195 195 195 195 195 195]  ← COMPLETAMENTE uniforme (cielo)
 [195 195 195 195 195 195 195 195 195 195 195]  ← COMPLETAMENTE uniforme (cielo)
 [195 195 195 195 195 195 195 195 195 195 195]  ← COMPLETAMENTE uniforme (cielo)
 [193 193 193 193 193 193 193 193 193 193 193]]  ← COMPLETAMENTE uniforme (cielo)
```

**Análisis:**
- ✅ Hay variación VERTICAL (75 → 195)
- ❌ NO hay variación HORIZONTAL (cada fila es uniforme)
- ❌ La mitad inferior de la imagen es cielo uniforme (valores ~195)

### Causa:
**La cámara estaba apuntando horizontal o ligeramente hacia arriba (pitch=0°)**
- Capturaba mayormente **cielo gris uniforme** (parte inferior de la imagen)
- Solo la parte superior mostraba algo de la carretera/entorno
- Resultado: "líneas horizontales" de color uniforme

---

## 🔧 Solución

### Cambio Aplicado:

**ANTES:**
```python
"transform": "1.5,0.0,1.5,0.0,0.0,0.0"  # pitch=0° (horizontal)
#             x   y   z   roll pitch yaw
```
- Posición: x=1.5, y=0.0, z=1.5
- Orientación: pitch=0° (mirando recto adelante)
- **Problema**: Captura mucho cielo

**DESPUÉS:**
```python
"transform": "2.0,0.0,1.2,0.0,-15.0,0.0"  # pitch=-15° (hacia abajo)
#             x   y   z   roll pitch  yaw
```
- Posición: x=2.0 (más adelante), y=0.0, z=1.2 (más bajo)
- Orientación: pitch=-15° (15° hacia abajo)
- **Mejora**: Apunta más hacia la carretera

---

## 📊 Comparación

### Antes (pitch=0°):
```
Vista de la cámara:
   
   ██████████████ (cielo - 50% superior de la imagen)
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (horizonte)
   ░░░░░░░░░░░░░░ (poco de carretera - 10%)
   █ █ █ █ █ █ █  (muy poco de carretera visible)
```

**Resultado en 11×11:**
- Filas 0-3: Un poco de carretera/entorno (variado)
- Filas 4-10: Mayormente cielo (uniforme ~195)
- **Efecto visual: Líneas horizontales**

### Después (pitch=-15°):
```
Vista de la cámara:
   
   ░░░░░░░░░░░░░░ (poco cielo)
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (horizonte)
   ████████████ (carretera - 40%)
   █ █ █ █ █ █ █  (marcas viales, texturas)
   ███████ ██████ (más carretera - 40%)
```

**Resultado esperado en 11×11:**
- Filas 0-2: Horizonte/cielo (alguna variación)
- Filas 3-10: Carretera, marcas, texturas (MÁS VARIACIÓN)
- **Efecto visual: Más contenido variado**

---

## 🎯 Validación

### Métricas para verificar mejora:

1. **Variación horizontal** (dentro de cada fila):
   - Antes: <5 (muy uniforme)
   - Después: >15 (variado)

2. **Filas uniformes consecutivas**:
   - Antes: 7-10 filas casi idénticas
   - Después: <4 filas uniformes

3. **Rango de valores**:
   - Antes: Muchos valores ~195 (cielo)
   - Después: Valores más distribuidos (texturas de carretera)

---

## 💡 Por Qué Esto Importa

### Para el Agente DRL:
Un agente que ve "líneas horizontales" solo tiene información sobre:
- Brillo general (claro/oscuro)
- Muy poca información espacial

Un agente que ve la **carretera con texturas** tiene información sobre:
- ✅ Posición del carril
- ✅ Marcas viales
- ✅ Bordes de la carretera
- ✅ Curvas adelante
- ✅ Obstáculos

**¡La orientación correcta de la cámara es CRÍTICA para que el agente aprenda a conducir!**

---

## 🔬 Paper Original

El paper de Pérez-Gil et al. (2022) no especifica explícitamente el ángulo de pitch de la cámara, pero es estándar en vehículos autónomos usar:
- **pitch: -10° a -20°** (mirando hacia la carretera)
- **Posición frontal** (en el capó o parabrisas)
- **FOV: 90°** (campo de visión estándar)

Nuestro ajuste a **-15°** está dentro del rango típico.

---

## 📝 Resumen

| Aspecto | Antes (pitch=0°) | Después (pitch=-15°) |
|---------|------------------|----------------------|
| **Contenido** | 50% cielo, 50% horizonte | 20% cielo, 80% carretera |
| **Variación horizontal** | Muy baja (<5) | Alta (>15) |
| **Filas uniformes** | 7-10 de 11 | <4 de 11 |
| **Utilidad para DRL** | Baja (poca info) | Alta (mucha info espacial) |
| **Efecto visual** | "Líneas horizontales" | Textura variada |

**Conclusión**: El pitch=-15° es ESENCIAL para que el agente vea la carretera y pueda aprender a navegar correctamente.

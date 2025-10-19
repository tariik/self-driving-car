#!/bin/bash
#
# Script para verificar y lanzar CARLA con visualización de waypoints
#

echo "════════════════════════════════════════════════════════════════════"
echo "🔍 VERIFICACIÓN DE VENTANA DE CARLA"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Verificar si CARLA está corriendo
if pgrep -f CarlaUE4 > /dev/null; then
    echo "✅ CARLA está corriendo"
    CARLA_PID=$(pgrep -f CarlaUE4)
    echo "   PID: $CARLA_PID"
else
    echo "❌ CARLA NO está corriendo"
    echo ""
    echo "⚠️  Inicia CARLA primero:"
    echo "   ./launch_carla.sh"
    exit 1
fi

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "🖥️  VERIFICANDO DISPLAY"
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Verificar DISPLAY
if [ -z "$DISPLAY" ]; then
    echo "❌ DISPLAY no está configurado"
    echo "   Necesitas ejecutar CARLA con display para ver los waypoints"
    echo ""
    echo "💡 Soluciones:"
    echo "   1. Si estás en SSH: usa X11 forwarding"
    echo "   2. Si estás local: verifica que X server esté corriendo"
    echo "   3. Usa VNC o escritorio remoto"
else
    echo "✅ DISPLAY configurado: $DISPLAY"
fi

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "🪟 VERIFICANDO VENTANAS"
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Verificar si hay ventanas de CARLA
if command -v xdotool &> /dev/null; then
    CARLA_WINDOWS=$(xdotool search --name "Carla" 2>/dev/null | wc -l)
    if [ "$CARLA_WINDOWS" -gt 0 ]; then
        echo "✅ Ventana de CARLA encontrada ($CARLA_WINDOWS ventana(s))"
    else
        echo "⚠️  No se encontró ventana de CARLA"
        echo "   CARLA puede estar corriendo sin display (headless)"
    fi
else
    echo "⚠️  xdotool no disponible, no se puede verificar ventanas"
fi

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "💡 PARA VER LOS WAYPOINTS:"
echo "────────────────────────────────────────────────────────────────────"
echo ""
echo "1. ✅ CARLA debe estar corriendo con ventana visible"
echo "2. ✅ Ejecuta: python test_waypoint_visualization.py"
echo "3. ✅ Busca la ventana de CARLA en tu pantalla"
echo "4. ✅ Los waypoints aparecerán en la vista del spectator"
echo ""
echo "🎨 BUSCA:"
echo "   🔵 Punto AZUL (START)"
echo "   🟢 Línea VERDE de waypoints"
echo "   🔴 Punto ROJO (GOAL)"
echo "   🟡 Punto AMARILLO (actual, se mueve)"
echo ""
echo "════════════════════════════════════════════════════════════════════"

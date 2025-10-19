#!/bin/bash

# Script para preparar y lanzar el entrenamiento con CARLA + Ventana de visualización
# Basado en el ejemplo manual_control.py que funciona correctamente
# Uso: ./start_training.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARLA_DIR="/home/tarekkhalfaoui/carla"

echo "════════════════════════════════════════════"
echo "  🚗 CARLA DRL Training Setup"
echo "════════════════════════════════════════════"
echo ""

# 1. Matar instancias previas de CARLA
echo "🔍 Paso 1/4: Limpiando instancias previas de CARLA..."
CARLA_PIDS=$(ps aux | grep "CarlaUE4-Linux-Shipping" | grep -v grep | awk '{print $2}')
if [ ! -z "$CARLA_PIDS" ]; then
    echo "   ⚠️  Instancias encontradas, cerrando..."
    for pid in $CARLA_PIDS; do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 2
    echo "   ✓ Instancias cerradas"
else
    echo "   ✓ No hay instancias previas"
fi
echo ""

# 2. Verificar entorno virtual
echo "🐍 Paso 2/4: Verificando entorno virtual..."
if [ ! -d "$SCRIPT_DIR/env" ]; then
    echo "   ❌ Error: No se encuentra el entorno virtual en $SCRIPT_DIR/env"
    echo "   Créalo con: python3 -m venv env && source env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$SCRIPT_DIR/env/bin/activate"

# Verificar pygame
if ! python -c "import pygame" 2>/dev/null; then
    echo "   ⚠️  Pygame no encontrado, instalando..."
    pip install pygame
fi
echo "   ✓ Entorno virtual activado: $VIRTUAL_ENV"
echo "   ✓ Python: $(which python)"
echo ""

# 3. Lanzar CARLA Server con DISPLAY
echo "🚀 Paso 3/4: Lanzando CARLA Server..."
echo "   - Display: :51.0"
echo "   - Quality: Low"
echo "   - Port: 3000"
echo "   - Mode: RenderOffScreen"

cd "$CARLA_DIR"
DISPLAY=:51.0 ./CarlaUE4.sh -RenderOffScreen -quality-level=low -carla-rpc-port=3000 > /tmp/carla.log 2>&1 &
CARLA_PID=$!

echo "   ⏳ Esperando inicio del servidor (10 segundos)..."
sleep 10

if ps -p $CARLA_PID > /dev/null; then
    echo "   ✓ CARLA Server está corriendo (PID: $CARLA_PID)"
else
    echo "   ❌ Error: CARLA no se inició correctamente"
    echo "   Ver log: tail /tmp/carla.log"
    exit 1
fi
echo ""

# 4. Información final
echo "✅ Paso 4/4: Sistema listo"
echo ""
echo "════════════════════════════════════════════"
echo "  📊 Estado del Sistema"
echo "════════════════════════════════════════════"
echo "CARLA Server:  ✓ Corriendo en puerto 3000"
echo "Python env:    ✓ $VIRTUAL_ENV"
echo "Display:       ✓ :51.0"
echo ""
echo "════════════════════════════════════════════"
echo "  🎮 Lanzando entrenamiento con ventana..."
echo "════════════════════════════════════════════"
echo ""

# Lanzar el entrenamiento con DISPLAY para que pygame funcione
cd "$SCRIPT_DIR/src"
DISPLAY=:51.0 python main.py

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ Entrenamiento completado"
echo "════════════════════════════════════════════"
echo ""
echo "Ver video: vlc ../training_video.mp4"
echo "Detener CARLA: bash ../close_carla.sh"
echo ""

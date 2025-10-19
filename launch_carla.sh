#!/bin/bash

# Script para lanzar CARLA Server
# Mata instancias anteriores y lanza nueva con configuración óptima
# Uso: ./launch_carla.sh

echo "🔍 Buscando instancias de CARLA existentes..."
CARLA_PIDS=$(ps aux | grep "CarlaUE4-Linux-Shipping" | grep -v grep | awk '{print $2}')

if [ ! -z "$CARLA_PIDS" ]; then
    echo "⚠️  Instancias de CARLA encontradas:"
    ps aux | grep "CarlaUE4-Linux-Shipping" | grep -v grep
    echo ""
    echo "🔫 Matando instancias anteriores..."
    for pid in $CARLA_PIDS; do
        echo "  Matando proceso PID: $pid"
        kill -9 $pid 2>/dev/null
    done
    echo "✓ Instancias cerradas"
    sleep 2
else
    echo "✓ No hay instancias previas"
fi

echo ""
echo "🚀 Lanzando CARLA Server..."
echo "   - Display: :51.0"
echo "   - Quality: Low"
echo "   - Port: 3000"
echo "   - Mode: RenderOffScreen"
echo ""

cd /home/tarekkhalfaoui/carla
DISPLAY=:51.0 ./CarlaUE4.sh -RenderOffScreen -quality-level=low -carla-rpc-port=3000 &

echo "⏳ Esperando inicio del servidor..."
sleep 8

echo ""
echo "🔍 Verificando estado..."
if ps aux | grep "CarlaUE4-Linux-Shipping" | grep -v grep > /dev/null; then
    echo "✅ CARLA Server está corriendo"
    echo ""
    ps aux | grep "CarlaUE4-Linux-Shipping" | grep -v grep
else
    echo "❌ Error: CARLA no se inició correctamente"
    exit 1
fi

echo ""
echo "✓ CARLA listo para usar en puerto 3000"

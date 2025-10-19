#!/bin/bash
# Script para cerrar todas las instancias de CARLA

echo "Buscando instancias de CARLA..."
carla_pids=$(ps aux | grep "CarlaUE4-Linux-Shipping" | grep -v grep | awk '{print $2}')

if [ -z "$carla_pids" ]; then
    echo "✓ No hay instancias de CARLA ejecutándose"
    exit 0
fi

echo "Instancias de CARLA encontradas:"
ps aux | grep "CarlaUE4-Linux-Shipping" | grep -v grep

echo ""
echo "Cerrando instancias de CARLA..."
for pid in $carla_pids; do
    echo "  Matando proceso con PID: $pid"
    kill -15 $pid  # Intenta cerrar gracefully primero
    sleep 1
    
    # Verificar si aún está corriendo
    if ps -p $pid > /dev/null 2>&1; then
        echo "  Forzando cierre del PID: $pid"
        kill -9 $pid
    fi
done

echo ""
echo "✓ Instancias de CARLA cerradas"

# Esperar un poco para que libere la memoria
sleep 2

# Verificar memoria GPU
echo ""
echo "Estado de la memoria GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "  Usado: %d MB / %d MB (%.1f%% libre)\n", $1, $2, 100-($1/$2*100)}'
fi

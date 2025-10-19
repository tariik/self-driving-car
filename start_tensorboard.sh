#!/bin/bash
# Script para iniciar TensorBoard y visualizar métricas de entrenamiento

echo "🚀 Iniciando TensorBoard..."
echo ""
echo "📊 Métricas disponibles:"
echo "   • Episode/Total_Reward - Reward acumulado por episodio"
echo "   • Episode/Length - Duración del episodio (steps)"
echo "   • State/Velocity_vt - Velocidad del vehículo"
echo "   • State/Distance_dt - Distancia al centro del carril"
echo "   • State/Angle_phi_t - Ángulo con el carril"
echo "   • Action/* - Throttle, Steering, Brake"
echo "   • Training/Epsilon - Tasa de exploración"
echo "   • Cumulative/Collision_Rate - Tasa de colisiones"
echo ""
echo "🌐 Abriendo TensorBoard en: http://localhost:6006"
echo ""
echo "ℹ️  Presiona Ctrl+C para detener"
echo ""

# Iniciar TensorBoard
tensorboard --logdir=runs --bind_all

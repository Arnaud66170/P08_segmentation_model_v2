#!/bin/bash

echo "🧹 Nettoyage manuel de la mémoire GPU en cours..."

# Affiche les processus utilisant le GPU
nvidia-smi

# Tente de tuer les processus liés au GPU (radical)
PIDS=$(fuser -v /dev/nvidia* 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/) print $i}' | sort -u)
if [ -z "$PIDS" ]; then
    echo "✅ Aucun processus GPU à tuer. Mémoire déjà propre."
else
    echo "⚠️ Processus à terminer : $PIDS"
    for pid in $PIDS; do
        echo "🛑 Killing PID $pid..."
        kill -9 $pid
    done
    echo "✅ Mémoire GPU libérée."
fi

# Vérifie à nouveau la mémoire GPU
echo "📊 État GPU après nettoyage :"
nvidia-smi

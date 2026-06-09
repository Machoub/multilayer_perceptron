#!/bin/bash

# Stop le script si une commande échoue
set -e

echo "🚀 Création de l'environnement virtuel avec uv..."
# uv crée automatiquement un dossier .venv très rapidement
uv venv .venv

echo "✅ Activation de l'environnement..."
# Détermine le chemin du script d'activation selon le système
if [[ "$OSTYPE" == "msys"* ]]; then
    ACTIVATE_SCRIPT=".venv/Scripts/activate"
else
    ACTIVATE_SCRIPT=".venv/bin/activate"
fi

# Active l'environnement virtuel pour le reste de l'exécution du script Bash
if [ -f "$ACTIVATE_SCRIPT" ]; then
    source "$ACTIVATE_SCRIPT"
else
    echo "❌ Erreur : Impossible de trouver le script d'activation à l'emplacement $ACTIVATE_SCRIPT"
    exit 1
fi

echo "📦 Installation des dépendances dans le venv avec uv..."
# On force uv à utiliser le pip du venv activé pour tout installer proprement dedans
uv pip install -r requirements.txt

echo "✨ Tout est prêt !"
echo "Pour activer ton venv manuellement dans ton terminal actuel :"
echo "source $ACTIVATE_SCRIPT"
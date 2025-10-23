#!/bin/bash

# Stop le script si une commande échoue
set -e

echo "🚀 Création de l'environnement virtuel..."
python3 -m venv .venv

echo "✅ Activation de l'environnement..."
# Active l'environnement en fonction du système
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source .venv/bin/activate
elif [[ "$OSTYPE" == "darwin"* ]]; then
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys"* ]]; then
    source .venv/Scripts/activate
else
    echo "⚠️  Système non reconnu, active manuellement le venv."
fi

echo "📦 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Tout est prêt !"
echo "Pour lancer ton programme :"
echo "source .venv/bin/activate  # ou .venv\\Scripts\\activate sous Windows"

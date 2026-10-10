#!/usr/bin/env bash

# Vai nella cartella dove si trova questo script
cd "$(dirname "$0")"

# Attiva il venv
source venv/bin/activate

# Lancia la GUI
python3 app.py

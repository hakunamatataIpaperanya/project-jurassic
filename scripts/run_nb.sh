#!/bin/sh
# Rebuild and execute the EDA notebook in place (outputs saved).
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/build_nb.py notebooks/01_eda_fossils.ipynb && .venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/01_eda_fossils.ipynb', as_version=4)
try:
    NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}}).execute()
finally:
    nbformat.write(nb, 'notebooks/01_eda_fossils.ipynb')
print('OK')
"

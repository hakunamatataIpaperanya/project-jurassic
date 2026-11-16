#!/bin/sh
# Rebuild and execute the Pangea features notebook in place (outputs saved).
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/build_pangea_nb.py notebooks/08_features_pangea.ipynb && .venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/08_features_pangea.ipynb', as_version=4)
try:
    NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}}).execute()
finally:
    nbformat.write(nb, 'notebooks/08_features_pangea.ipynb')
print('OK')
"

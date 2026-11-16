#!/bin/sh
# Rebuild and execute the final-data EDA notebook in place (outputs saved).
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/build_eda_final_nb.py notebooks/06_eda_final.ipynb && .venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/06_eda_final.ipynb', as_version=4)
try:
    NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}}).execute()
finally:
    nbformat.write(nb, 'notebooks/06_eda_final.ipynb')
print('OK')
"

#!/bin/sh
# Rebuild and execute the model-prep imputation notebook in place (outputs saved).
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/build_impute_nb.py notebooks/07_impute_for_model.ipynb && .venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/07_impute_for_model.ipynb', as_version=4)
try:
    NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}}).execute()
finally:
    nbformat.write(nb, 'notebooks/07_impute_for_model.ipynb')
print('OK')
"

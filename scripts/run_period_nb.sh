#!/bin/sh
# Rebuild and execute the period-fill notebook in place (outputs saved).
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/build_period_nb.py notebooks/03_fill_period_pbdb.ipynb && .venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/03_fill_period_pbdb.ipynb', as_version=4)
try:
    NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}}).execute()
finally:
    nbformat.write(nb, 'notebooks/03_fill_period_pbdb.ipynb')
print('OK')
"

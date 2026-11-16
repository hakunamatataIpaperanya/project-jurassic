#!/bin/sh
# Rebuild and execute the taxon period-fill notebook in place (outputs saved).
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/build_taxon_fill_nb.py notebooks/04_fill_period_taxon.ipynb && .venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/04_fill_period_taxon.ipynb', as_version=4)
try:
    NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}}).execute()
finally:
    nbformat.write(nb, 'notebooks/04_fill_period_taxon.ipynb')
print('OK')
"

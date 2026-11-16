#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
caffeinate -i .venv/bin/python scripts/run_notebook.py build_task1_nb.py 11_task1_models.ipynb 21600

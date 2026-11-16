#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
caffeinate -i .venv/bin/python scripts/run_notebook.py build_task1_fe_nb.py 15_task1_fe.ipynb 28800

#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
caffeinate -i .venv/bin/python scripts/run_notebook.py build_task2_fe_nb.py 14_task2_fe.ipynb 28800

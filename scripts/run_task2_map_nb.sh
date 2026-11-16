#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
caffeinate -i .venv/bin/python scripts/run_notebook.py build_task2_map_nb.py 10b_task2_map.ipynb 21600

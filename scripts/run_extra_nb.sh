#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
caffeinate -i .venv/bin/python scripts/run_notebook.py build_extra_nb.py 13_features_extra.ipynb 14400

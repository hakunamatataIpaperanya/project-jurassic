# Rebuild a notebook with its builder and execute it in place (outputs saved even if a cell fails).
# usage: .venv/bin/python scripts/run_notebook.py <builder.py> <notebook.ipynb> [cell_timeout_seconds]
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nb_clean import clean_notebook

root = Path(__file__).resolve().parent.parent
builder, nb_path = sys.argv[1], sys.argv[2]
timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 1800
subprocess.run([sys.executable, str(root / "scripts" / builder), str(root / "notebooks" / nb_path)], check=True, cwd=root)
nb = nbformat.read(root / "notebooks" / nb_path, as_version=4)
try:
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(root / "notebooks")}}).execute()
finally:
    nbformat.write(clean_notebook(nb), root / "notebooks" / nb_path)
print("OK")

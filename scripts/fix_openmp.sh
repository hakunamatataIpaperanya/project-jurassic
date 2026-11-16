#!/bin/sh
# Make LightGBM and XGBoost use PyTorch's bundled libomp so only ONE OpenMP runtime is loaded per process.
# (Homebrew libomp + torch libomp in one process -> segfault / deadlock on macOS.)
# Idempotent. Re-run after reinstalling lightgbm, xgboost or torch.
set -e
SP="$(dirname "$0")/../.venv/lib/python3.12/site-packages"
for lib in "$SP/lightgbm/lib/lib_lightgbm.dylib" "$SP/xgboost/lib/libxgboost.dylib"; do
  cur=$(otool -L "$lib" | awk '/libomp.dylib/ {print $1}')
  if [ "$cur" != "@loader_path/../../torch/lib/libomp.dylib" ]; then
    install_name_tool -change "$cur" "@loader_path/../../torch/lib/libomp.dylib" "$lib"
    codesign --force -s - "$lib"
    echo "patched $(basename "$lib"): $cur -> torch libomp"
  else
    echo "ok $(basename "$lib")"
  fi
done

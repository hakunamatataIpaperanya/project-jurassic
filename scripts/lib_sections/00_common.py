# Shared paths, constants and small helpers for all feature groups.
import json
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3",
                  GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="15", GDAL_CACHEMAX="2048",
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", VSI_CACHE="TRUE", VSI_CACHE_SIZE="536870912",
                  GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2")

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
DATA = ROOT / "data"
FEAT = DATA / "features"
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
THAI_BBOX = (97.3, 5.6, 105.7, 20.5)


def tqdm_auto():
    from tqdm.auto import tqdm
    return tqdm

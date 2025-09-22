import logging
import os
from functools import partial

import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

my_tqdm = partial(tqdm.tqdm, mininterval=20, maxinterval=60)

# ref:
# https://github.com/facebookresearch/stopes/blob/main/stopes/modules/bitext/indexing/train_faiss_index_module.py#L55
from pathlib import Path

import faiss
import numpy as np
from stopes.modules.bitext.indexing.train_index import train_index

from svecalign.utils.log_utils import logging

logger = logging.getLogger(__name__)


def train_faiss_index(
        embedding_file: Path,
        index_type: str,
        use_gpu: bool,
        out_dir: Path,
        fp16: bool = False,
        embedding_dimensions: int = 1024
) -> Path:
    index_output_file = (
            out_dir / f"{index_type}.train.idx"
    ).resolve()

    returned_index = train_index(
        train_embeddings=embedding_file,
        idx_type=index_type,
        dim=embedding_dimensions,
        gpu=use_gpu,
        dtype=np.float16 if fp16 else np.float32,
    )

    if use_gpu:
        returned_index = faiss.index_gpu_to_cpu(returned_index)
    faiss.write_index(returned_index, str(index_output_file))

    logger.info(
        f"Trained index of type: {index_type}, can be found in output file: {index_output_file}"
    )

    return index_output_file

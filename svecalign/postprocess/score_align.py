# this script computes margin-scores for alignments with trained indices.
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stopes.modules.bitext.mining.calculate_distances_utils import load_index  # noqa

from svecalign.postprocess.prep_index import load_embed_from_tsv
from svecalign.utils.file_utils import read_alignments, read_metadata
from svecalign.utils.log_utils import logging, my_tqdm

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata", type=str,
        help="the meta file that each line contains paired audio paths"
    )
    parser.add_argument(
        "out_dir", type=str,
        help="dir to store the sampled embeddings, and indices."
    )
    parser.add_argument(
        "--embed_dir", type=str, required=True,
        help="the dir for embedding tsvs."
    )
    parser.add_argument(
        "--align_dir", type=str, required=True,
        help="the dir for concatenated alignments."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
    )
    # index-related
    parser.add_argument(
        "--index_dir", type=str, required=True,
        help="where the indexes are saved."
    )
    parser.add_argument(
        "--num_probe", type=int, default=128
    )
    parser.add_argument(
        "--gpu_type", type=str, default="fp16-shard"
    )
    # embedding related
    parser.add_argument(
        "--embed_fp16", action="store_true", default=False,
        help="whether the embeddings are saved in fp16."
    )
    parser.add_argument(
        "--embed_stopes", action="store_true", default=False,
        help="whether the input embeddings are saved with stopes."
    )
    # margin related
    parser.add_argument(
        "--margin", type=str, default="ratio",
        help="Margin for xSIM calculation. See: https://aclanthology.org/P19-1309",
    )
    parser.add_argument(
        "--k", type=int, default=16,
        help="number of nearest number."
    )
    return parser.parse_args()


def find_valid_metas(
        meta: List[Tuple[str, str]],
        embed_dir: Path,
) -> List[str]:
    res = []
    for src_aud, tgt_aud in meta:
        src_id = Path(src_aud).stem
        tgt_id = Path(tgt_aud).stem
        src_tsv = embed_dir / f"{src_id}-{tgt_id}.src.tsv"
        tgt_tsv = embed_dir / f"{src_id}-{tgt_id}.tgt.tsv"

        if src_tsv.exists() and tgt_tsv.exists():
            res.append(
                f"{src_id}-{tgt_id}"
            )
        elif not src_tsv.exists() and not tgt_tsv.exists():
            logger.warning(f"{src_tsv} and {tgt_tsv} not exist")
        else:
            raise Exception(f"{src_tsv}: {src_tsv.exists()} | {tgt_tsv}: {tgt_tsv.exists()}")

    logger.info(f"Kept {len(res)}/{len(meta)}")
    return res


def write_to_output(
        align_dir: Path,
        align_ids: List[str],
        margin_scores: np.ndarray,
        out_dir: Path
):
    # counter for margin scores
    margin_id = 0

    # must follow exactly the order
    for ali_id in align_ids:
        align_path = align_dir / f"{ali_id}.txt"
        alignments = read_alignments(align_path)

        with open(out_dir / f"{ali_id}.txt", mode="w") as fp:
            for src, tgt in alignments:
                fp.write(f"{src}:{tgt}:{margin_scores[margin_id]}\n")
                margin_id += 1

    assert margin_id == margin_scores.shape[0], f"{margin_id}, {margin_scores.shape}"


def inplace_l2_to_cosine(x: np.ndarray):
    np.negative(x, out=x)
    np.add(x, 2, out=x)
    np.divide(x, 2.0, out=x)


def compute_sim_with_nonflat_idx(
        idx_x, idx_y,
        x: np.ndarray, y: np.ndarray,
        k: int, margin: str,
) -> np.ndarray:
    import faiss  # noqa

    num_x, dim_x = x.shape
    num_y, dim_y = y.shape
    assert num_x == num_y and dim_x == dim_y, f"{x.shape} {y.shape}"

    faiss.normalize_L2(x)
    faiss.normalize_L2(y)

    # L2 square between x and its k nearest neighbors in y
    L2_square_xy, Idx_xy = idx_y.search(x, k)  # num_x, k
    # L2 square between y and its k nearest neighbors in x
    L2_square_yx, Idx_yx = idx_x.search(y, k)  # num_y, k

    # average L2 squares
    Avg_xy = L2_square_xy.mean(axis=1)
    Avg_yx = L2_square_yx.mean(axis=1)

    # In-place L2 square to cosine: cosine = (2 - L2^2) / 2
    inplace_l2_to_cosine(Avg_xy)
    inplace_l2_to_cosine(Avg_yx)

    scores = np.zeros(shape=(num_x,), dtype=np.float32)
    for i in range(num_x):
        a = np.dot(x[i], y[i])
        b = (Avg_xy[i] + Avg_yx[i]) / 2
        if margin == "ratio":
            scores[i] = a / b
        elif margin == "distance":
            scores[i] = a - b
        else:
            raise ValueError(f"Wrong margin type: {margin}")
    return scores


def main():
    args = parse_args()
    logger.info(args)

    # inputs
    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    embed_fp16 = args.embed_fp16
    embed_stopes = args.embed_stopes
    logger.info(f"fp16: {embed_fp16} | stopes: {embed_stopes}")

    margin = args.margin
    k = args.k
    logger.info(f"margin: {margin} | k: {k}")

    src_lang, tgt_lang = args.src_lang, args.tgt_lang

    embed_dir = Path(args.embed_dir) / f"{src_lang}-{tgt_lang}"
    align_dir = Path(args.align_dir) / f"{src_lang}-{tgt_lang}"

    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0. find valid pairs
    metas = find_valid_metas(all_pairs, embed_dir)
    del all_pairs

    # 1. load index
    index_dir = Path(args.index_dir) / f"{src_lang}-{tgt_lang}"

    src_index_path = list((index_dir / src_lang).glob(f"*.populate.idx"))[0]
    src_index_type = src_index_path.name.split(".")[0]

    tgt_index_path = list((index_dir / tgt_lang).glob(f"*.populate.idx"))[0]
    tgt_index_type = tgt_index_path.name.split(".")[0]

    num_probe = args.num_probe
    gpu_type = args.gpu_type

    logger.info(f"num_probe: {num_probe} | gpu_type: {gpu_type}")
    logger.info(f"Loading {src_index_path} as {src_index_type}")
    src_index = load_index(
        idx_name=src_index_path.as_posix(),
        nprobe=num_probe,
        gpu_type=gpu_type,
        index_type=src_index_type,
    )
    logger.info(f"Loading {tgt_index_path} as {tgt_index_type}")
    tgt_index = load_index(
        idx_name=tgt_index_path.as_posix(),
        nprobe=num_probe,
        gpu_type=gpu_type,
        index_type=tgt_index_type,
    )

    # calculate scores
    margin_scores = []
    for align_id in my_tqdm(metas):
        src_embed = load_embed_from_tsv(
            embed_dir / f"{align_id}.src.tsv",
            fp16_embed=embed_fp16, use_stopes=embed_stopes
        )
        tgt_embed = load_embed_from_tsv(
            embed_dir / f"{align_id}.tgt.tsv",
            fp16_embed=embed_fp16, use_stopes=embed_stopes
        )
        margin_scores.append(
            compute_sim_with_nonflat_idx(
                src_index, tgt_index,
                src_embed, tgt_embed,
                k, margin
            )
        )
    margin_scores = np.concatenate(margin_scores, axis=0)

    logger.info(f"Writing to {out_dir}...")
    write_to_output(
        align_dir,
        metas,
        margin_scores,
        out_dir
    )
    logger.info(f"Done!")


if __name__ == '__main__':
    main()

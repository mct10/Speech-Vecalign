# This script consists of the following steps:
#   1. Randomly sample a subset of embeddings can concatenate them into a single file.
#   2. Train an index on the sampled embedding file.
#   3. Add all embeddings into the trained index.
# Inputs:
#   Should be ${src_id}-${tgt_id}.src.tsv, ${src_id}-${tgt_id}.tgt.tsv.
# See embed_align.py for format.
import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import faiss  # noqa
import numpy as np
from stopes.core.utils import count_lines  # noqa
from stopes.modules.bitext.indexing.train_index import index_to_gpu  # noqa
from stopes.utils.embedding_utils import Embedding, EmbeddingConcatenator  # noqa
from stopes.utils.mining_utils import determine_faiss_index_type  # noqa

from svecalign.utils.embedding_utils import load_sent_embeddings
from svecalign.utils.file_utils import delete_if_exist, read_metadata
from svecalign.utils.log_utils import logging, my_tqdm
from svecalign.utils.mining_utils import train_faiss_index

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
        "--data_dir", type=str, required=True,
        help="the dir for embedding tsvs."
    )
    parser.add_argument(
        "--use_tgt", action="store_true", default=False,
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=0.5,
        help="percentage of embedding files used for training indexes."
    )
    parser.add_argument(
        "--embed_fp16", action="store_true", default=False,
        help="whether the embeddings are saved in fp16."
    )
    parser.add_argument(
        "--embed_stopes", action="store_true", default=False,
        help="whether the input embeddings are saved with stopes."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
    )
    return parser.parse_args()


def find_embed_files(
        meta: List[Tuple[str, str]],
        data_dir: Path,
        use_tgt: bool
) -> List[Path]:
    res = []
    for src_aud, tgt_aud in meta:
        src_id = Path(src_aud).stem
        tgt_id = Path(tgt_aud).stem
        src_tsv = data_dir / f"{src_id}-{tgt_id}.src.tsv"
        tgt_tsv = data_dir / f"{src_id}-{tgt_id}.tgt.tsv"

        if src_tsv.exists() and tgt_tsv.exists():
            res.append(
                tgt_tsv if use_tgt else src_tsv
            )
        elif not src_tsv.exists() and not tgt_tsv.exists():
            logger.warning(f"{src_tsv} and {tgt_tsv} do not exist")
        else:
            raise Exception(f"{src_tsv}: {src_tsv.exists()} | {tgt_tsv}: {tgt_tsv.exists()}")
    logger.info(f"Kept {len(res)}/{len(meta)} files")
    return res


def load_embed_from_tsv(
        tsv_path: Path,
        fp16_embed: bool,
        use_stopes: bool
) -> np.ndarray:
    """
    tsv_path contains:
    path_to_embed_file index
    """
    # key=embed_file
    info_dict = defaultdict(list)
    with open(tsv_path) as fp:
        for ii, line in enumerate(fp):
            path, _id = line.strip().split("\t")
            info_dict[path].append(
                (ii, int(_id))
            )

    # find ids for each embed_path
    true_ids = []
    embeds = []
    for embed_path in info_dict.keys():
        embed = load_sent_embeddings(
            embed_path,
            fp16_embed=fp16_embed,
            use_stopes=use_stopes,
            stopes_mode="memory"
        )
        for true_id, embed_id in info_dict[embed_path]:
            true_ids.append(true_id)
            embeds.append(embed[embed_id])

    # concat
    true_ids = np.argsort(true_ids)
    embeds = np.stack(embeds)[true_ids]
    return embeds


def dump_embedding_to_file(
        embed_paths: List[Path],
        out_path: Path,
        fp16_embed: bool,
        use_stopes: bool
) -> Tuple[int, Path]:
    delete_if_exist(out_path, verbose=True)

    n_lines = 0
    with EmbeddingConcatenator(out_path, fp16_embed) as combined_fp:
        for path in my_tqdm(embed_paths):
            embed = load_embed_from_tsv(
                path,
                fp16_embed=fp16_embed,
                use_stopes=use_stopes
            )
            combined_fp.append_embedding_from_array(embed)

            n_lines += count_lines(path)

        logger.info(combined_fp.shape)
    return n_lines, out_path.resolve()


def populate_index(
        index_path: Path,
        embed_paths: List[Path],
        out_path: Path,
        gpu: bool,
        fp16_embed: bool,
        use_stopes: bool
):
    """
    Add embeddings to the index.
    """
    index = faiss.read_index(index_path.as_posix())

    if gpu:
        index = index_to_gpu(index)

    for path in my_tqdm(embed_paths):
        embed = load_embed_from_tsv(
            path,
            fp16_embed=fp16_embed,
            use_stopes=use_stopes
        )
        # TODO: embeddings in each file should not be too much,
        # so currently we do not use chunking
        faiss.normalize_L2(embed)
        index.add(embed)

    if gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(
        index,
        out_path.as_posix(),
    )


def main():
    args = parse_args()
    logger.info(args)

    # inputs
    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    embed_fp16 = args.embed_fp16
    embed_stopes = args.embed_stopes
    logger.info(f"fp16: {embed_fp16} | stopes: {embed_stopes}")

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    data_dir = Path(args.data_dir) / f"{src_lang}-{tgt_lang}"

    use_tgt = args.use_tgt
    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    if use_tgt:
        out_dir = out_dir / tgt_lang
    else:
        out_dir = out_dir / src_lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0. find valid inputs
    # this should follow the order of metadata
    embed_paths = find_embed_files(all_pairs, data_dir, use_tgt)
    del all_pairs

    # 1. prepare training samples
    sample_ratio = args.sample_ratio
    sample_size = int(sample_ratio * len(embed_paths))
    sample_size = max(sample_size, 1)  # at least choose one file!
    logger.info(f"Will sample {sample_size}/{len(embed_paths)} files.")
    training_samples = random.Random(42).sample(embed_paths, k=sample_size)  # fix seed
    logger.info(f"Examples: {training_samples[:5]}")

    n_samples, sample_embed_path = dump_embedding_to_file(
        training_samples,
        out_path=out_dir / f"sample.embed",
        fp16_embed=embed_fp16,
        use_stopes=embed_stopes
    )
    logger.info(f"Sampled {n_samples} embeddings.")

    # 2. train
    n_embed_tot = 0
    for _path in embed_paths:
        n_embed_tot += count_lines(_path)
    logger.info(f"#embeddings: {n_embed_tot}")

    logger.info(f"Training...")
    index_type = determine_faiss_index_type(n_embed_tot)
    trained_index = train_faiss_index(
        embedding_file=sample_embed_path,
        index_type=index_type,
        use_gpu=True,
        out_dir=out_dir,
        fp16=embed_fp16,
        embedding_dimensions=1024
    )
    logger.info(f"Dumped index to {trained_index}")

    # 3. add embeddings to index
    populate_index(
        trained_index,
        embed_paths,
        out_path=out_dir / f"{index_type}.populate.idx",
        gpu=True,
        fp16_embed=embed_fp16,
        use_stopes=embed_stopes
    )
    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

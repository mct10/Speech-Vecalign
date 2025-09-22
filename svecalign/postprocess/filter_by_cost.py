import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional

from svecalign.utils.file_utils import read_metadata, read_alignments_with_score
from svecalign.utils.log_utils import my_tqdm, logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata", type=str,
        help="the meta file that each line contains paired audio paths"
    )
    parser.add_argument(
        "out_dir", type=str,
        help="dir to save alignments."
    )
    parser.add_argument(
        "--align_dir", type=str, required=True,
        help="where the alignments are saved."
    )
    parser.add_argument(
        "--max_cost", type=float, required=True,
        help="the threshold."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
    )
    return parser.parse_args()


def do_filter(
        in_path: str,
        out_path: str,
        max_cost: Optional[float] = None,
        min_cost: Optional[float] = None
) -> float:
    """
    A general function that supports min and max filtering.
    """
    assert (max_cost is None and min_cost is not None) or (max_cost is not None and min_cost is None), \
        f"{min_cost} {max_cost}"

    old_cnt = new_cnt = 0
    low_quality_cnt = deletion_cnt = 0

    # just in case all segments are filtered out
    in_alignments = read_alignments_with_score(in_path)
    out_alignments = []
    for src_segs, tgt_segs, cost in in_alignments:
        old_cnt += 1
        # filter deletions
        if len(src_segs) == 0 or len(tgt_segs) == 0:
            deletion_cnt += 1
            continue
        # filter bad alignments
        if max_cost is not None and cost > max_cost:
            low_quality_cnt += 1
            continue
        if min_cost is not None and cost < min_cost:
            low_quality_cnt += 1
            continue

        # write to output
        new_cnt += 1
        out_alignments.append((src_segs, tgt_segs, cost))

    if out_alignments:
        with open(out_path, mode="w") as out_fp:
            for src_segs, tgt_segs, cost in out_alignments:
                out_fp.write(f"{src_segs}:{tgt_segs}:{cost}\n")
    else:
        logger.warning(f"Empty output. Will not write!")

    # it is recommended to check this
    logger.debug(f"{os.path.basename(in_path)} "
                 f"|| Threshold: {max_cost} || #Kept: {new_cnt}/{old_cnt} "
                 f"|| #Low quality: {low_quality_cnt} || #Deletions: {deletion_cnt}")

    return new_cnt / old_cnt


def main():
    args = parse_args()
    logger.info(args)

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    src_lang, tgt_lang = args.src_lang, args.tgt_lang

    align_dir = Path(args.align_dir) / f"{src_lang}-{tgt_lang}"
    max_cost = args.max_cost

    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_alignments = []

    for src, tgt in my_tqdm(all_pairs):
        src_stem = Path(src).stem
        tgt_stem = Path(tgt).stem
        kept_ratio = do_filter(
            in_path=(align_dir / f"{src_stem}-{tgt_stem}.txt").as_posix(),
            out_path=(out_dir / f"{src_stem}-{tgt_stem}.txt").as_posix(),
            max_cost=max_cost
        )
        if kept_ratio < 0.5:
            bad_alignments.append(f"{src_stem}-{tgt_stem}")

    logger.info(f"{len(bad_alignments)} / {len(all_pairs)} pairs kept less than half alignments.")
    logger.debug(bad_alignments)
    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

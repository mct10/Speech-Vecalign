"""
This script converts the Vecalign alignment files into a tsv file,
so that we can prepare the training process using that tsv file.

Input alignment file:
  [7, 8, 9]:[7, 8, 9]:0.95855
Output tsv file:
  0.95855 ${src_aud_path} 247328 297952 16    ${tgt_aud_path} 314400 356832 16
  score \t src_path src_s src_e 16 \t tgt_path tgt_s tgt_e 16
"""

import argparse
from pathlib import Path
from typing import List, Union, Tuple

import svecalign.utils.file_utils as file_utils
from svecalign.utils.file_utils import read_segments, alignments_to_timestamps, read_alignments_with_score
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
        help="output dir of the tsv file."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
    )
    parser.add_argument(
        "--align_dir", type=str, required=True,
        help="dir to all alignments."
    )
    parser.add_argument(
        "--seg_dir", type=str, required=True,
        help="dir for original segments."
    )
    return parser.parse_args()


def make_meta(
        align_path: Union[Path, str],
        src_seg_path: Union[Path, str],
        tgt_seg_path: Union[Path, str],
        src_audio_path: str,
        tgt_audio_path: str,
) -> List[Tuple[float, str]]:
    """
    Prepare meta for one alignment file.
    May contain more than one alignment.
    """

    src_segs = read_segments(src_seg_path)
    tgt_segs = read_segments(tgt_seg_path)

    src_frames, tgt_frames, tot = alignments_to_timestamps(
        align_path, src_segs, tgt_segs,
        ignore_empty=False
    )

    alignments = read_alignments_with_score(align_path)
    assert len(src_frames) == len(tgt_frames) == len(alignments)

    res = []
    for i in range(tot):
        score = float(alignments[i][2])
        src_info = f"{src_audio_path} {src_frames[i][0]} {src_frames[i][1]} 16"
        tgt_info = f"{tgt_audio_path} {tgt_frames[i][0]} {tgt_frames[i][1]} 16"
        res.append(
            (score, f"{score}" + "\t" + src_info + "\t" + tgt_info)
        )
    return res


def main():
    args = parse_args()
    logger.info(args)

    # inputs
    all_pairs: List[Tuple[str, str]] = file_utils.read_metadata(args.metadata)
    src_lang, tgt_lang = args.src_lang, args.tgt_lang

    align_dir = Path(args.align_dir) / f"{src_lang}-{tgt_lang}"
    src_seg_dir = Path(args.seg_dir) / src_lang
    tgt_seg_dir = Path(args.seg_dir) / tgt_lang

    # output
    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "align.tsv.gz"
    assert not out_path.exists(), f"{out_path} exists. Will not overwrite."

    # load all alignments
    meta_collections = []
    for src_aud_path, tgt_aud_path in my_tqdm(all_pairs):
        src_stem = Path(src_aud_path).stem
        tgt_stem = Path(tgt_aud_path).stem

        # get paths
        align_path = align_dir / f"{src_stem}-{tgt_stem}.txt"
        if not align_path.exists():
            logger.warning(f"{align_path} not exist. Skip.")
            continue

        # retrieve alignments from a single file
        tsv_meta = make_meta(
            align_path,
            src_seg_dir / f"{src_stem}.txt",
            tgt_seg_dir / f"{tgt_stem}.txt",
            src_aud_path, tgt_aud_path
        )
        meta_collections.extend(tsv_meta)

    # highest margin-score first
    meta_collections.sort(key=lambda x: -x[0])

    with file_utils.open(out_path, mode="w") as fp:
        for _, line in meta_collections:
            fp.write(line + "\n")
    logger.info("Finished!")


if __name__ == '__main__':
    main()

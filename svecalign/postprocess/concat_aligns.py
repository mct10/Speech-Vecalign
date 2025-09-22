import argparse
from pathlib import Path
from typing import List, Tuple

from svecalign.utils.audio_utils import SAMPLE_RATE
from svecalign.utils.file_utils import read_alignments, read_segments, read_metadata, write_alignment
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
        help="where to save the concatenated alignments."
    )
    parser.add_argument(
        "--max_num_align", type=int,
        help="max num of consecutive alignments to be concatenated."
    )
    parser.add_argument(
        "--align_dir", type=str, required=True,
        help="where the alignments are saved."
    )
    parser.add_argument(
        "--seg_dir", type=str, required=True,
        help="the dir for all segments."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
    )
    parser.add_argument(
        "--max_sil", type=float, default=1.0,
        help="if silence in between is longer than this, DO NOT concatenate."
    )
    parser.add_argument(
        "--max_dur", type=float, default=20.0,
        help="if the sum of two consecutive segment durations is larger than this, then no grouping."
    )
    parser.add_argument(
        "--apply_dur_cond_to_both_sides", action="store_true", default=False,
        help="Whether to apply `max_dur` to both sides. "
             "By default (False), only applied to the source side."
    )
    return parser.parse_args()


def group_aligns_by_num(
        alignments: List[Tuple[List[int], List[int]]],
        src_seg_to_frames: List[Tuple[int, int]],
        tgt_seg_to_frames: List[Tuple[int, int]],
        max_num_align: int,
        max_sil: float,
        max_dur: float,
        sample_rate: int,
        apply_dur_cond_to_both_sides: bool = False
) -> List[Tuple[List[int], List[int]]]:
    assert max_num_align >= 1, max_num_align

    res: List[Tuple[List[int], List[int]]] = []

    for start_i in range(len(alignments)):
        _src, _tgt = alignments[start_i]

        this_src = _src + []
        this_tgt = _tgt + []
        res.append((this_src, this_tgt))  # always add original alignments

        for step in range(1, max_num_align):
            end_i = start_i + step
            if end_i > len(alignments) - 1:
                break

            next_src, next_tgt = alignments[end_i]
            # 0. dur check
            src_dur = (src_seg_to_frames[next_src[-1]][1] - src_seg_to_frames[this_src[0]][0]) / sample_rate
            if src_dur > max_dur:
                break

            tgt_dur = (tgt_seg_to_frames[next_tgt[-1]][1] - tgt_seg_to_frames[this_tgt[0]][0]) / sample_rate
            if apply_dur_cond_to_both_sides and tgt_dur > max_dur:
                break

            # 1. connected
            if not (next_src[0] == this_src[-1] + 1 and next_tgt[0] == this_tgt[-1] + 1):
                break

            # 2. silence
            src_sil = (src_seg_to_frames[next_src[0]][0] - src_seg_to_frames[this_src[-1]][1]) / sample_rate
            tgt_sil = (tgt_seg_to_frames[next_tgt[0]][0] - tgt_seg_to_frames[this_tgt[-1]][1]) / sample_rate

            if src_sil > max_sil or tgt_sil > max_sil:
                break

            # concat
            next_src = this_src + next_src
            next_tgt = this_tgt + next_tgt
            res.append((next_src, next_tgt))

            this_src = next_src
            this_tgt = next_tgt
    return res


def main():
    args = parse_args()
    logger.info(args)

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    # hyperparameters
    max_num_align = args.max_num_align
    max_sil = args.max_sil
    max_dur = args.max_dur
    apply_dur_cond_to_both_sides = args.apply_dur_cond_to_both_sides
    logger.info(f"max_num_align: {max_num_align} | "
                f"max_sil: {max_sil} | "
                f"max_dur: {max_dur} (to both sides? {apply_dur_cond_to_both_sides})")

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    align_dir = Path(args.align_dir) / f"{src_lang}-{tgt_lang}"
    src_seg_dir = Path(args.seg_dir) / src_lang
    tgt_seg_dir = Path(args.seg_dir) / tgt_lang

    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src_audio, tgt_audio in my_tqdm(all_pairs):
        src_stem = Path(src_audio).stem
        tgt_stem = Path(tgt_audio).stem

        in_align_path = align_dir / f"{src_stem}-{tgt_stem}.txt"
        if not in_align_path.exists():
            logger.warning(f"{in_align_path.as_posix()} not exist. Skip.")
            continue

        raw_alignments = read_alignments(in_align_path)
        if len(raw_alignments) == 0:
            logger.warning(f"{in_align_path.as_posix()} is empty. Skip.")
            continue

        src_segs = read_segments(src_seg_dir / f"{src_stem}.txt")
        tgt_segs = read_segments(tgt_seg_dir / f"{tgt_stem}.txt")

        grouped_alignments = group_aligns_by_num(
            alignments=raw_alignments,
            src_seg_to_frames=src_segs, tgt_seg_to_frames=tgt_segs,
            max_num_align=max_num_align,
            max_sil=max_sil, max_dur=max_dur,
            sample_rate=SAMPLE_RATE,
            apply_dur_cond_to_both_sides=apply_dur_cond_to_both_sides
        )

        write_alignment(grouped_alignments, out_dir / f"{src_stem}-{tgt_stem}.txt")
    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

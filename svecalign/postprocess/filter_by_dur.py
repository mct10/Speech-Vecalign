import argparse
from pathlib import Path
from typing import Tuple, List

from svecalign.utils.audio_utils import SAMPLE_RATE
from svecalign.utils.file_utils import read_lines, read_segments, alignments_to_timestamps, read_metadata
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
        help="dir to save alignments."
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
        "--min_dur", type=float, default=1.0,
        help="alignments shorter than this will be discarded. in second."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
    )
    return parser.parse_args()


def do_filter(
        align_path: Path,
        src_seg_path: Path,
        tgt_seg_path: Path,
        min_frames: int,
        out_path: Path
):
    src_segments = read_segments(src_seg_path)
    tgt_segments = read_segments(tgt_seg_path)
    src_frames, tgt_frames, cnt = alignments_to_timestamps(
        align_path,
        src_segments, tgt_segments,
        ignore_empty=True
    )

    res = []
    alignments = read_lines(align_path)
    for ii in range(cnt):
        # BOTH sides
        if min_frames <= src_frames[ii][1] - src_frames[ii][0] and \
                min_frames <= tgt_frames[ii][1] - tgt_frames[ii][0]:
            res.append(alignments[ii])

    if len(res) == 0:
        logger.info(f"Skip {out_path.as_posix()}. You can double check inputs {align_path.as_posix()}")
    else:
        with open(out_path, mode="w") as fp:
            for line in res:
                fp.write(line + "\n")


def main():
    args = parse_args()
    logger.info(args)

    # inputs
    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    src_lang, tgt_lang = args.src_lang, args.tgt_lang

    align_dir = Path(args.align_dir) / f"{src_lang}-{tgt_lang}"
    seg_dir = Path(args.seg_dir)
    min_frames = int(SAMPLE_RATE * args.min_dur)
    logger.info(f"Min frames: {min_frames}")

    # outputs
    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src, tgt in my_tqdm(all_pairs):
        src_stem = Path(src).stem
        tgt_stem = Path(tgt).stem

        in_align_path = align_dir / f"{src_stem}-{tgt_stem}.txt"
        if not in_align_path.exists():
            logger.warning(f"{in_align_path.as_posix()} not exist. Skip.")
            continue

        do_filter(
            in_align_path,
            seg_dir / src_lang / f"{src_stem}.txt",
            seg_dir / tgt_lang / f"{tgt_stem}.txt",
            min_frames,
            out_dir / f"{src_stem}-{tgt_stem}.txt",
        )
    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

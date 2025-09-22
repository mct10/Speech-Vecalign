import argparse
from pathlib import Path
from typing import List, Tuple, Union, Set

from svecalign.utils.audio_utils import SAMPLE_RATE
from svecalign.utils.file_utils import read_metadata, read_segments
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
        help="the output dir."
    )
    parser.add_argument(
        "--seg_dir", type=str, required=True,
        help="dir saving all segments."
    )
    parser.add_argument(
        "--identical_seg_dir", type=str, required=True,
        help="dir saving all untranslated identical segment ids."
    )
    parser.add_argument(
        "--src_lang",
        required=True,
        type=str,
        help="the language code."
    )
    parser.add_argument(
        "--tgt_lang",
        required=True,
        type=str,
        help="the language code."
    )
    parser.add_argument(
        "--num_overlaps",
        type=int,
        default=5,
        help="Maximum number of allowed overlaps."
    )
    parser.add_argument(
        "--max_dur",
        type=float,
        default=20.0,
        help="Maximum time each concatenated segment can have. (in seconds)"
    )
    return parser.parse_args()


def load_indices(path: Union[str, Path]) -> Set[int]:
    res = set()
    with open(path) as fp:
        for line in fp:
            res.add(int(line.strip()))
    return res


def get_identical_overlap_ids(
        in_path: Union[str, Path],
        num_overlaps: int,
        max_frames: int,
        identical_segs_path: Union[str, Path]
) -> List[Tuple[int, int]]:
    """
    This basically reruns `overlap`.
    """
    assert num_overlaps > 0, num_overlaps

    segs = read_segments(in_path)
    identical_segs = load_indices(identical_segs_path)

    ignore_indices = []

    for i, (start, end) in enumerate(segs):
        # a single segment could also exceed max frames
        if end - start > max_frames:
            continue

        # this single one should be ignored.
        if i in identical_segs:
            ignore_indices.append((i, i))
            continue

        for j in range(1, num_overlaps):
            # out of bound
            if i + j >= len(segs):
                break
            # too long
            if segs[i + j][1] - start > max_frames:
                break
            # the newly added segment should be ignored
            if (i + j) in identical_segs:
                ignore_indices.append((i, i + j))
                break
    return ignore_indices


def main():
    args = parse_args()
    logger.info(args)

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    # input dirs
    seg_dir = Path(args.seg_dir)
    identical_seg_dir = Path(args.identical_seg_dir) / f"{src_lang}-{tgt_lang}"

    # overlap configs
    num_overlaps, max_dur = args.num_overlaps, args.max_dur

    # output dir
    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src_audio_path, tgt_audio_path in my_tqdm(all_pairs):
        src_stem = Path(src_audio_path).stem
        tgt_stem = Path(tgt_audio_path).stem

        src_ignore_indices = get_identical_overlap_ids(
            in_path=seg_dir / src_lang / f"{src_stem}.txt",
            num_overlaps=num_overlaps,
            max_frames=int(max_dur * SAMPLE_RATE),
            identical_segs_path=identical_seg_dir / f"{src_stem}-{tgt_stem}.src.txt"
        )

        tgt_ignore_indices = get_identical_overlap_ids(
            in_path=(seg_dir / tgt_lang / f"{tgt_stem}.txt").as_posix(),
            num_overlaps=num_overlaps,
            max_frames=int(max_dur * SAMPLE_RATE),
            identical_segs_path=identical_seg_dir / f"{src_stem}-{tgt_stem}.tgt.txt"
        )

        with open(out_dir / f"{src_stem}-{tgt_stem}.src.txt", mode="w") as fp:
            for i, j in src_ignore_indices:
                fp.write(f"{i} {j}\n")
        with open(out_dir / f"{src_stem}-{tgt_stem}.tgt.txt", mode="w") as fp:
            for i, j in tgt_ignore_indices:
                fp.write(f"{i} {j}\n")

    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

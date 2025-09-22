import argparse
from pathlib import Path
from typing import Tuple, List

from svecalign.utils.audio_utils import SAMPLE_RATE, find_untranslated_segs
from svecalign.utils.file_utils import read_metadata, read_segments
from svecalign.utils.log_utils import logging, my_tqdm
from svecalign.utils.mp_utils import start_multi_processes

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
        help="the dir for all segments."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
        help="use for seg_dir/src_lang and out_dir/src_lang-tgt_lang"
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
        help="use for seg_dir/tgt_lang and out_dir/src_lang-tgt_lang"
    )
    # criteria for identical untranslated segments
    parser.add_argument(
        "--dur_diff", type=float,
        default=0.1,  # 10frames * 10ms = 0.1s
        help="the max time difference between two segments. in second"
    )
    parser.add_argument(
        "--fbank_dist_thres", type=float, default=5.0,
        help="the MSE threshold between two fbank."
    )
    parser.add_argument(
        "--n_proc", type=int, default=1,
        help="Num of processes."
    )
    return parser.parse_args()


def detect(
        pid: int,
        pairs: List[Tuple[str, str]],
        src_seg_dir: Path, tgt_seg_dir: Path,
        max_frame_diff: int,
        fbank_dist_thres: float,
        out_dir: Path
):
    for src_audio_path, tgt_audio_path in my_tqdm(pairs, desc=f"[Proc {pid}]"):
        src_name = Path(src_audio_path).stem
        tgt_name = Path(tgt_audio_path).stem

        src_out_path = out_dir / f"{src_name}-{tgt_name}.src.txt"
        tgt_out_path = out_dir / f"{src_name}-{tgt_name}.tgt.txt"

        # skip processed inputs
        if src_out_path.exists() and tgt_out_path.exists():
            continue

        src_segs = read_segments(src_seg_dir / f"{src_name}.txt")
        tgt_segs = read_segments(tgt_seg_dir / f"{tgt_name}.txt")

        duplicates = find_untranslated_segs(
            src_segs, tgt_segs,
            src_audio_path, tgt_audio_path,
            max_frame_diff, fbank_dist_thres
        )

        # note, still will write an empty file if there are no identical segments
        tmp_src_out_path = out_dir / f"{src_name}-{tgt_name}.src.txt.tmp"
        tmp_tgt_out_path = out_dir / f"{src_name}-{tgt_name}.tgt.txt.tmp"
        with open(tmp_src_out_path, mode="w") as src_fp, \
                open(tmp_tgt_out_path, mode="w") as tgt_fp:
            for _src_dup, _tgt_dup in duplicates:
                src_fp.write(f"{_src_dup}\n")
                tgt_fp.write(f"{_tgt_dup}\n")
        tmp_src_out_path.replace(src_out_path)
        tmp_tgt_out_path.replace(tgt_out_path)


def main():
    args = parse_args()
    logger.info(args)

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_frame_diff = int(args.dur_diff * SAMPLE_RATE)  # sec -> frames

    start_multi_processes(
        data=all_pairs,
        n_proc=args.n_proc,
        func=detect,
        src_seg_dir=Path(args.seg_dir) / src_lang,
        tgt_seg_dir=Path(args.seg_dir) / tgt_lang,
        max_frame_diff=max_frame_diff, fbank_dist_thres=args.fbank_dist_thres,
        out_dir=out_dir
    )
    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

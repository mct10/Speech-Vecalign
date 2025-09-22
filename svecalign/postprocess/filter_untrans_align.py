import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Union

from svecalign.utils.audio_utils import SAMPLE_RATE, Segment, compute_fbank_dist
from svecalign.utils.file_utils import read_lines, delete_if_exist, read_segments, read_metadata, read_alignments, \
    alignments_to_timestamps
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
        help="dir to save cleaned alignments."
    )
    parser.add_argument(
        "--align_dir", type=str, required=True,
        help="where the alignments are saved."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
    )
    parser.add_argument(
        "--seg_dir", type=str, required=True,
        help="the dir for all segments."
    )
    parser.add_argument(
        "--dur_diff", type=float,
        default=0.5,  # 0.5s -> 25 frames
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
    parser.add_argument(
        "--save_audio", action="store_true", default=False,
        help="Whether to save the untranslated alignments as audios."
             "If True, will save to `${out_dir}/wavs`."
    )
    return parser.parse_args()


def check_and_save(
        align_path: Union[str, Path],
        src_segs: List[Tuple[int, int]],
        tgt_segs: List[Tuple[int, int]],
        src_audio_path: Union[str, Path],
        tgt_audio_path: Union[str, Path],
        max_frame_diff: int,
        fbank_dist_thres: float,
        out_path: Path,
        audio_out_dir: Optional[Path] = None,
        use_gpu: bool = False
) -> int:
    """
    Check the segments in `src_segs` and `tgt_segs` in pairs.
    If `audio_out_dir` is not None, save audios to it.
    """
    duplicate_cnt = 0

    alignments = read_alignments(align_path)

    src_times, tgt_times, n_samples = alignments_to_timestamps(
        align=alignments,
        src_segs=src_segs, tgt_segs=tgt_segs,
        ignore_empty=False
    )
    assert n_samples == len(alignments)

    align_id_to_save = []
    for ii in range(n_samples):
        src_seg = Segment(start=src_times[ii][0], end=src_times[ii][1], path=src_audio_path)
        tgt_seg = Segment(start=tgt_times[ii][0], end=tgt_times[ii][1], path=tgt_audio_path)

        if abs(src_seg.duration - tgt_seg.duration) > max_frame_diff:
            align_id_to_save.append(ii)
            continue

        dist = compute_fbank_dist(src_seg.fbank(use_gpu), tgt_seg.fbank(use_gpu))
        if dist > fbank_dist_thres:
            align_id_to_save.append(ii)
            continue

        # a duplicate detected!
        duplicate_cnt += 1

        if audio_out_dir:
            if not audio_out_dir.exists():
                audio_out_dir.mkdir(parents=True, exist_ok=True)
            # save audio for debug. save as ogg to save space.
            src_seg.save((audio_out_dir / f"{ii}.src.ogg").as_posix())
            tgt_seg.save((audio_out_dir / f"{ii}.tgt.ogg").as_posix())

    if len(align_id_to_save) == 0:
        logger.info(f"{align_path} is completely filtered out.")
    else:
        # write the clean alignments to output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = read_lines(align_path)
        with open(out_path, mode="w") as fp:
            for ii in align_id_to_save:
                fp.write(f"{lines[ii]}\n")

    return duplicate_cnt


def check_alignments(
        pid: int,
        audio_pairs: List[Tuple[str, str]],
        align_dir: Path,
        seg_dir: Path,
        src_lang: str, tgt_lang: str,
        max_frame_diff: int,
        fbank_dist_thres: float,
        out_dir: Path,
        audio_out_dir: Optional[Path] = None
):
    duplicate_cnt = 0

    for src_audio, tgt_audio in my_tqdm(audio_pairs, desc=f"[Proc {pid}]"):
        src_stem = Path(src_audio).stem
        tgt_stem = Path(tgt_audio).stem

        # there is no input
        in_align_path = align_dir / f"{src_stem}-{tgt_stem}.txt"
        if not in_align_path.exists():
            logger.warning(f"{in_align_path.as_posix()} not exist. Skip.")
            continue

        # skip processed
        out_align_path = out_dir / f"{src_stem}-{tgt_stem}.txt"
        if out_align_path.exists():
            continue

        tmp_out_align_path = out_align_path.with_suffix(".tmp")
        delete_if_exist(tmp_out_align_path)

        src_segs = read_segments(
            seg_dir / src_lang / f"{src_stem}.txt"
        )
        tgt_segs = read_segments(
            seg_dir / tgt_lang / f"{tgt_stem}.txt"
        )

        _cnt = check_and_save(
            align_path=in_align_path,
            src_segs=src_segs, tgt_segs=tgt_segs,
            src_audio_path=src_audio, tgt_audio_path=tgt_audio,
            max_frame_diff=max_frame_diff,
            fbank_dist_thres=fbank_dist_thres,
            out_path=tmp_out_align_path,
            audio_out_dir=audio_out_dir / f"{src_stem}-{tgt_stem}" if audio_out_dir is not None \
                else None
        )

        tmp_out_align_path.replace(out_align_path)
        duplicate_cnt += _cnt

    logger.info(f"Found {duplicate_cnt} duplications!")


def main():
    args = parse_args()
    logger.info(args)

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    max_frame_diff = int(args.dur_diff * SAMPLE_RATE)  # sec -> frames

    src_lang, tgt_lang = args.src_lang, args.tgt_lang

    align_dir = Path(args.align_dir) / f"{src_lang}-{tgt_lang}"

    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_audio = args.save_audio
    if save_audio:
        audio_out_dir = out_dir / "wavs"
        audio_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        audio_out_dir = None

    start_multi_processes(
        data=all_pairs,
        n_proc=args.n_proc,
        func=check_alignments,
        align_dir=align_dir,
        seg_dir=Path(args.seg_dir),
        src_lang=src_lang, tgt_lang=tgt_lang,
        max_frame_diff=max_frame_diff,
        fbank_dist_thres=args.fbank_dist_thres,
        out_dir=out_dir,
        audio_out_dir=audio_out_dir
    )


if __name__ == '__main__':
    main()

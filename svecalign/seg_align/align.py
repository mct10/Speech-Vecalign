import argparse
import dataclasses
from pathlib import Path
from typing import List, Tuple, Optional, Union

from svecalign.utils.file_utils import read_metadata, check_exist
from svecalign.utils.log_utils import logging, my_tqdm
from svecalign.vecalign.vecalign import align as vecalign_func

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
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
    )
    parser.add_argument(
        "--seg_dir", type=str, required=True,
        help="the dir for raw segments."
    )
    parser.add_argument(
        "--concat_dir", type=str, required=True,
        help="the dir for concatenated segments."
    )
    parser.add_argument(
        "--embed_dir", type=str, required=True,
        help="Dir to embedding files."
    )
    parser.add_argument(
        "--is_stopes_embed", action="store_true", default=False,
        help="whether the embeddings are dumped by stopes. Used for SpeechLASER."
    )
    parser.add_argument(
        "--fp16_embed", action="store_true", default=False,
        help="whether the embeddings are saved as fp16. used for SONAR (numpy) embeddings"
    )
    # vecalign args
    parser.add_argument(
        '-a', '--alignment_max_size', dest="alignment_max_size",
        type=int,
        default=6,
        help='Searches for alignments up to size N-M, where N+M <= this value.'
             'Note that the the embeddings must support the requested number of overlaps'
    )
    parser.add_argument(
        '--search_buffer_size',
        type=int,
        default=5,
        help='Width (one side) of search buffer. '
             'Larger values makes search more likely to recover from errors but increases runtime.'
    )
    parser.add_argument(
        '-d', '--del_percentile_frac', dest="del_percentile_frac",
        type=float,
        default=0.2,
        help='Deletion penalty is set to this percentile (as a fraction) of the cost matrix distribution. '
             'Should be between 0 and 1.'
    )
    parser.add_argument(
        '--max_size_full_dp',
        type=int,
        default=300,
        help='Maximum size N for which is is acceptable to run full N^2 dynamic programming.'
    )
    parser.add_argument(
        '--costs_sample_size',
        type=int,
        default=20000,
        help='Sample size to estimate costs distribution, used to set deletion penalty in conjunction with deletion_percentile.'
    )
    parser.add_argument(
        '--num_samps_for_norm',
        type=int,
        default=100,
        help='Number of samples used for normalizing embeddings'
    )
    # used to ignore identical untranslated segments
    parser.add_argument(
        "--ign_indices_dir",
        type=str,
        default=None,
        help="if provided, then some segments will be ignored when loading embeddings."
    )
    return parser.parse_args()


@dataclasses.dataclass
class VecalignData:
    src_seg_path: str
    tgt_seg_path: str

    src_concat_path: str
    tgt_concat_path: str

    src_embed_path: str
    tgt_embed_path: str

    output_path: str

    # some optional fields
    src_ignore_indices: Optional[Union[str, Path]] = None
    tgt_ignore_indices: Optional[Union[str, Path]] = None


def validate_inputs(
        audio_pairs: List[Tuple[str, str]],
        src_seg_dir: Path, tgt_seg_dir: Path,
        src_concat_dir: Path, tgt_concat_dir: Path,
        src_embed_dir: Path, tgt_embed_dir: Path,
        out_dir: Path,
        ign_indices_dir: Optional[Path] = None
) -> List[VecalignData]:
    """
    To make sure all the required files exists.
    Also to pack data into a more handy format...
    """
    res = []
    for src_audio, tgt_audio in audio_pairs:
        src_name = Path(src_audio).name
        tgt_name = Path(tgt_audio).name
        src_stem = Path(src_audio).stem
        tgt_stem = Path(tgt_audio).stem

        # segments
        src_seg_path = (src_seg_dir / src_name).with_suffix(".txt")
        tgt_seg_path = (tgt_seg_dir / tgt_name).with_suffix(".txt")
        if not check_exist(src_seg_path) or not check_exist(tgt_seg_path):
            continue

        # concatenated seg
        src_concat_path = (src_concat_dir / src_name).with_suffix(".txt")
        tgt_concat_path = (tgt_concat_dir / tgt_name).with_suffix(".txt")
        if not check_exist(src_concat_path) or not check_exist(tgt_concat_path):
            continue

        # embeddings
        src_embed_path = (src_embed_dir / src_name).with_suffix(".embed")
        tgt_embed_path = (tgt_embed_dir / tgt_name).with_suffix(".embed")
        if not check_exist(src_embed_path) or not check_exist(tgt_embed_path):
            continue

        # ignore segments
        if ign_indices_dir is None:
            src_ign_ind_path = tgt_ign_ind_path = None
        else:
            src_ign_ind_path = ign_indices_dir / f"{src_stem}-{tgt_stem}.src.txt"
            tgt_ign_ind_path = ign_indices_dir / f"{src_stem}-{tgt_stem}.tgt.txt"

            if not check_exist(src_ign_ind_path):
                src_ign_ind_path = None
            if not check_exist(tgt_ign_ind_path):
                tgt_ign_ind_path = None

        res.append(
            VecalignData(
                src_seg_path=src_seg_path.as_posix(),
                tgt_seg_path=tgt_seg_path.as_posix(),
                src_concat_path=src_concat_path.as_posix(),
                tgt_concat_path=tgt_concat_path.as_posix(),
                src_embed_path=src_embed_path.as_posix(),
                tgt_embed_path=tgt_embed_path.as_posix(),
                output_path=(out_dir / f"{src_stem}-{tgt_stem}.txt").as_posix(),
                src_ignore_indices=src_ign_ind_path,
                tgt_ignore_indices=tgt_ign_ind_path
            )
        )
    return res


def main():
    args = parse_args()
    logger.info(args)

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ign_indices_dir is None:
        ign_indices_dir = None
    else:
        ign_indices_dir = Path(args.ign_indices_dir) / f"{src_lang}-{tgt_lang}"
        logger.info(f"Will ignore segments indicated by {ign_indices_dir}")

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)
    valid_pairs = validate_inputs(
        all_pairs,
        Path(args.seg_dir) / src_lang, Path(args.seg_dir) / tgt_lang,
        Path(args.concat_dir) / src_lang, Path(args.concat_dir) / tgt_lang,
        Path(args.embed_dir) / src_lang, Path(args.embed_dir) / tgt_lang,
        out_dir,
        ign_indices_dir
    )

    for pair in my_tqdm(valid_pairs):
        pair: VecalignData
        vecalign_func(
            src=pair.src_seg_path, tgt=pair.tgt_seg_path,
            src_embed=[pair.src_concat_path, pair.src_embed_path],
            src_stopes=args.is_stopes_embed,
            src_fp16=args.fp16_embed,
            tgt_embed=[pair.tgt_concat_path, pair.tgt_embed_path],
            tgt_stopes=args.is_stopes_embed,
            tgt_fp16=args.fp16_embed,
            alignment_max_size=args.alignment_max_size,
            many_to_one=None,
            search_buffer_size=args.search_buffer_size,
            del_percentile_frac=args.del_percentile_frac,
            max_size_full_dp=args.max_size_full_dp,
            costs_sample_size=args.costs_sample_size,
            num_samps_for_norm=args.num_samps_for_norm,
            overlap_segments=True,
            print_aligned_text=False,
            print_results=True,
            save_aligned_text_to_file=pair.output_path,
            verbose=False,
            src_ignore_indices=pair.src_ignore_indices,
            tgt_ignore_indices=pair.tgt_ignore_indices
        )


if __name__ == '__main__':
    main()

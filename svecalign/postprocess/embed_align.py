"""
This script embeds the alignments.
Output format:
1. One tsv file, ${src_id}-${tgt_id}.{src|tgt}.tsv, either for source or for target. With columns:
  path    id
e.g.,
  a.embed b
meaning that, the embedding for the first line of alignment is at a.embed[b].
The path could be any embedding. The goal is to reuse the existing embeddings.
2. (Opt) One embedding file, ${src_id}-${tgt_id}.{src|tgt}.embed. Either for source or for target.
They are generated if the embeddings do not exist in previously computed embeddings.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Tuple, Dict, List, Union

from svecalign.utils.embed_model_utils import embed_to_file, load_embed_model, add_embed_args, \
    save_segment_audio_and_tsv
from svecalign.utils.file_utils import delete_if_exist, read_segments, read_metadata, \
    alignments_to_timestamps
from svecalign.utils.log_utils import logging, my_tqdm

logger = logging.getLogger(__name__)

PID = str(os.getpid())
logger.info(f"Pid: {PID}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata", type=str,
        help="the meta file that each line contains paired audio paths"
    )
    parser.add_argument(
        "out_dir", type=str,
        help="where to save the embeddings and tsvs."
    )
    parser.add_argument(
        "--src_lang", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_lang", type=str, required=True,
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
        "--concat_seg_dir", type=str, required=True,
        help="dir for all concatenated segments"
    )
    parser.add_argument(
        "--concat_seg_embed_dir", type=str, required=True,
        help="dir for all concatenated segments' embeddings"
    )
    parser.add_argument(
        "--use_tgt", action="store_true", default=False,
        help="whether to embed target side."
    )

    add_embed_args(parser)

    return parser.parse_args()


def find_reusable_embeddings(
        all_segments: List[Tuple[int, int]],
        overlap_seg_path: Path
) -> Tuple[Dict[int, int], List[int]]:
    """
    Given all the segments (`all_segments`) and the embedded segments (`overlap_seg_path`),
        return (1) the mapping between ids in all_segments to overlap_seg_path;
            (2) the ids that still need embedding.
    """
    existed_segments = read_segments(overlap_seg_path)
    existed_segments_to_id = {
        _seg: _id for _id, _seg in enumerate(existed_segments)
    }

    id_mapping = {}
    miss_ids = []
    for ii, seg in enumerate(all_segments):
        if seg in existed_segments_to_id:
            id_mapping[ii] = existed_segments_to_id[seg]
        else:
            miss_ids.append(ii)
    return id_mapping, miss_ids


def load_one_side_alignments(
        align_path: Union[str, Path],
        src_seg_path: Union[str, Path],
        tgt_seg_path: Union[str, Path],
        embed_source: bool
) -> List[Tuple[int, int]]:
    """
    Load either source or target, not both.
    """
    src_segs = read_segments(src_seg_path)
    tgt_segs = read_segments(tgt_seg_path)

    src_aligns, tgt_aligns, n_aligns = alignments_to_timestamps(
        align_path, src_segs, tgt_segs,
        ignore_empty=False
    )

    if embed_source:
        return src_aligns
    else:
        return tgt_aligns


def main():
    args = parse_args()
    logger.info(args)

    # inputs
    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)

    embed_src = not args.use_tgt

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    embed_lang = src_lang if embed_src else tgt_lang
    file_suffix = "src" if embed_src else "tgt"

    logger.info(f"Will process {'src' if embed_src else 'tgt'}. Lang={embed_lang}. Suffix={file_suffix}")

    align_dir = Path(args.align_dir) / f"{src_lang}-{tgt_lang}"
    seg_dir = Path(args.seg_dir)
    concat_seg_dir = Path(args.concat_seg_dir) / embed_lang
    concat_seg_embed_dir = Path(args.concat_seg_embed_dir) / embed_lang

    # outputs
    out_dir = Path(args.out_dir) / f"{src_lang}-{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load embedding model
    embed_model_type = args.embed_model_type
    logger.info(f"Embed model = {embed_model_type}")
    embed_model = load_embed_model(
        embed_model_type,
        sl_ckpt_dir=args.sl_ckpt_dir, sl_ckpt_name=args.sl_ckpt_name, max_tokens=args.max_tokens,
        sonar_name=args.sonar_name, sonar_fp16=not args.embed_fp32, compile_sonar=args.compile_sonar
    )

    for src_aud_path, tgt_aud_path in my_tqdm(all_pairs):
        src_stem = Path(src_aud_path).stem
        tgt_stem = Path(tgt_aud_path).stem

        embed_stem = src_stem if embed_src else tgt_stem
        embed_aud_path = src_aud_path if embed_src else tgt_aud_path

        # 0. setup output files. will override these.
        tsv_out_path = out_dir / f"{src_stem}-{tgt_stem}.{file_suffix}.tsv"
        embed_out_path = out_dir / f"{src_stem}-{tgt_stem}.{file_suffix}.embed"
        # tsv file alone is enough to judge whether processed or not
        if tsv_out_path.exists():
            continue

        delete_if_exist(tsv_out_path)
        delete_if_exist(embed_out_path)

        # 1. find segments needs embedding
        in_align_path = align_dir / f"{src_stem}-{tgt_stem}.txt"
        if not in_align_path.exists():
            logger.warning(f"{in_align_path.as_posix()} not exist. Skip.")
            continue

        all_segments = load_one_side_alignments(
            in_align_path,
            src_seg_path=seg_dir / src_lang / f"{src_stem}.txt",
            tgt_seg_path=seg_dir / tgt_lang / f"{tgt_stem}.txt",
            embed_source=embed_src
        )

        # 2. find existing embeddings
        reuse_seg_id_to_overlap_embed_id, miss_seg_ids = find_reusable_embeddings(
            all_segments, concat_seg_dir / f"{embed_stem}.txt"
        )
        logger.info(f"{src_stem}-{tgt_stem}: "
                    f"n_hit={len(reuse_seg_id_to_overlap_embed_id)} | n_miss={len(miss_seg_ids)}")

        # 3. embed new segments (if there are any)
        if len(miss_seg_ids) > 0:
            miss_segments = [all_segments[_id] for _id in miss_seg_ids]
            tmp_embed_out_path = embed_out_path.with_suffix(".tmp")
            delete_if_exist(tmp_embed_out_path)

            with tempfile.TemporaryDirectory(prefix=PID) as _tmp_dir:
                tmp_dir = Path(_tmp_dir)
                save_segment_audio_and_tsv(
                    out_dir=tmp_dir,
                    wave_path=embed_aud_path,
                    segments=miss_segments,
                )
                embed_to_file(
                    embed_model,
                    embed_model_type,
                    tmp_dir,
                    tmp_embed_out_path,
                    fp16=not args.embed_fp32,
                    batch_size=args.batch_size,
                    n_proc=args.n_proc
                )
            tmp_embed_out_path.replace(embed_out_path)

        # 4. output
        overlap_embed_path = concat_seg_embed_dir / f"{embed_stem}.embed"
        miss_seg_to_embed_id = None
        if miss_seg_ids:
            miss_seg_to_embed_id = {all_segments[_id]: ii for ii, _id in enumerate(miss_seg_ids)}

        tmp_tsv_out_path = tsv_out_path.with_suffix(".tmp")
        delete_if_exist(tmp_tsv_out_path)
        with open(tmp_tsv_out_path, mode="w") as fp:
            for ii, seg in enumerate(all_segments):
                if ii in reuse_seg_id_to_overlap_embed_id:
                    fp.write(
                        f"{overlap_embed_path.as_posix()}" + "\t" + f"{reuse_seg_id_to_overlap_embed_id[ii]}" + "\n"
                    )
                else:
                    fp.write(
                        f"{embed_out_path.as_posix()}" + "\t" + f"{miss_seg_to_embed_id[seg]}" + "\n"
                    )
        tmp_tsv_out_path.replace(tsv_out_path)

    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

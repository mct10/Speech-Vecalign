import argparse
import tempfile
from pathlib import Path
from typing import Tuple, List

from svecalign.utils.embed_model_utils import add_embed_args, load_embed_model, save_segment_audio_and_tsv, \
    embed_to_file
from svecalign.utils.file_utils import read_metadata, read_segments
from svecalign.utils.log_utils import logging, my_tqdm
from svecalign.utils.mp_utils import get_shard_range

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata", type=str,
        help="the meta file that each line contains paired audio paths"
    )
    parser.add_argument(
        "out_dir", type=str,
        help="the output dir. will save as *.embed."
    )
    parser.add_argument(
        "--concat_dir", type=str, required=True,
        help="will apply embedding to all segments files (*.txt) here."
    )
    parser.add_argument(
        "--lang", type=str, required=True,
        help="language. read from `concat_dir/lang`, output to `out_dir/lang`."
    )
    parser.add_argument(
        "--use_tgt", action="store_true", default=False,
        help="whether to read target side."
    )
    # sharding
    parser.add_argument(
        "--rank",
        type=int,
        default=0
    )
    parser.add_argument(
        "--n_shard",
        type=int,
        default=1
    )

    add_embed_args(parser)

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)
    if args.use_tgt:
        all_inputs = [pair[1] for pair in all_pairs]
    else:
        all_inputs = [pair[0] for pair in all_pairs]
    all_inputs = sorted(list(set(all_inputs)))  # unique, sorted list

    start, end = get_shard_range(len(all_inputs), args.n_shard, args.rank)
    all_inputs = all_inputs[start:end]

    # input segments
    lang = args.lang
    concat_dir = Path(args.concat_dir) / lang

    # output dir
    out_dir = Path(args.out_dir) / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_model_type = args.embed_model_type
    logger.info(f"Embed model = {embed_model_type}")
    embed_model = load_embed_model(
        embed_model_type,
        sl_ckpt_dir=args.sl_ckpt_dir, sl_ckpt_name=args.sl_ckpt_name, max_tokens=args.max_tokens,
        sonar_name=args.sonar_name
    )

    embed_fp32 = args.embed_fp32
    logger.info(f"Embed with {'fp32' if embed_fp32 else 'fp16'}")

    for audio_file in my_tqdm(all_inputs):
        audio_stem = Path(audio_file).stem
        # make sure there are available segments
        seg_file = concat_dir / f"{audio_stem}.txt"
        if not seg_file.exists():
            logger.warning(f"{seg_file} not exists! Skip.")
            continue

        all_segments = read_segments(seg_file)
        if len(all_segments) == 0:
            logger.warning(f"encountered empty segment file {seg_file}, corresponding audio file is {audio_file}")
            continue

        # prepare segment audios and tsv in /tmp
        tmp_out_path = out_dir / f"{audio_stem}.tmp.embed"
        embed_out_path = out_dir / f"{audio_stem}.embed"

        # skip processed inputs
        if embed_out_path.exists():
            continue

        with tempfile.TemporaryDirectory() as _tmp_dir:
            tmp_dir = Path(_tmp_dir)
            save_segment_audio_and_tsv(
                out_dir=tmp_dir,
                wave_path=audio_file,
                segments=all_segments,
            )
            embed_to_file(
                embed_model,
                embed_model_type,
                tmp_dir,
                tmp_out_path,
                fp16=not embed_fp32,
                batch_size=args.batch_size,
                n_proc=args.n_proc
            )
        tmp_out_path.replace(embed_out_path)
    logger.info("Finished!")


if __name__ == '__main__':
    main()

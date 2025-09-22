# ref:https://github.com/facebookresearch/stopes/blob/main/stopes/modules/speech/postprocess.py
# just a wrapper to bypass the config
import argparse
from pathlib import Path

from stopes.modules.speech.postprocess import PostProcessAudioConfig, PostProcessAudioModule  # noqa

from svecalign.utils.log_utils import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, required=True,
    )
    parser.add_argument(
        "--output_filename", type=str, required=True,
    )
    parser.add_argument(
        "--mining_result_path", type=Path, required=True,
        help="The input alignment tsv file."
    )
    parser.add_argument(
        "--min_audio_length", type=int, required=True,
        help="Audio longer than this will not be loaded. In millisecond."
    )
    parser.add_argument(
        "--mining_threshold", type=float, required=True,
        help="Alignments whose scores lower than this will not be loaded."
    )
    parser.add_argument(
        "--max_overlap", type=float, default=0.2,
        help="The maximum overlap ratio."
    )
    return parser.parse_args()


def run(
        output_dir: Path,
        output_filename: str,
        mining_result_path: Path,
        min_audio_length: int,
        mining_threshold: float,
        max_overlap: float = 0.2,  # max admissible overlap see,
):
    assert not (output_dir / output_filename).exists(), \
        f"The output path {output_dir / output_filename} already exists!"

    config = PostProcessAudioConfig(
        output_dir=output_dir,
        output_filename=output_filename,
        mining_result_path=mining_result_path,
        min_audio_length=min_audio_length,
        mining_threshold=mining_threshold,
        max_overlap=max_overlap,
    )

    module = PostProcessAudioModule(config)
    outfile = module.run()
    logger.info(f"Output to {outfile}")


if __name__ == '__main__':
    _args = parse_args()
    logger.info(_args)
    run(**vars(_args))

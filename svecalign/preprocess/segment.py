"""
Apply silero-VAD to each speech file.
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch

from svecalign.utils.audio_utils import SAMPLE_RATE
from svecalign.utils.file_utils import read_metadata, check_exist
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
        help="base output directory."
    )
    parser.add_argument(
        "--lang", type=str, required=True,
        help="output segments to `out_dir/lang`"
    )
    parser.add_argument(
        "--use_tgt", default=False, action="store_true",
        help="whether to read target side."
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="which shard this job will process. range: [0, n_shard)."
    )
    parser.add_argument(
        "--n_shard", type=int, default=1,
        help="number of shards in total."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Dir to save the VAD model."
    )
    parser.add_argument(
        "--vad_version", type=str, default="snakers4/silero-vad",
        help="Which version of silero vad to use. By default the latest one will be used."
             "We used `snakers4/silero-vad:v4.0` for the paper."
    )
    return parser.parse_args()


class VadModel:
    """
    A wrapper for silero-vad.
    """

    def __init__(self, vad_version: str, cache_dir: Optional[str] = None):
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(cache_dir)

        model, utils = torch.hub.load(
            repo_or_dir=vad_version,
            model='silero_vad',
            force_reload=False,
            onnx=False
        )

        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils

        self.__model = model
        self.__silero_save_audio_func = save_audio
        self.__silero_read_audio_func = read_audio
        self.__get_speech_timestamps = get_speech_timestamps

    def silero_read_audio(self, wav: str, sampling_rate: int) -> torch.Tensor:
        return self.__silero_read_audio_func(wav, sampling_rate)

    def silero_save_audio(self, path, tensor, **kwargs):
        self.__silero_save_audio_func(path=path, tensor=tensor, **kwargs)

    def __call__(self, waveform: torch.Tensor, sampling_rate: int, **kwargs):
        return self.__get_speech_timestamps(waveform, self.__model, sampling_rate=sampling_rate, **kwargs)


def vad(
        vad_version: str,
        file_paths: List[str],
        output_dir: Path,
        cache_dir: Optional[str] = None,
):
    vad_model = VadModel(vad_version=vad_version, cache_dir=cache_dir)
    for in_path in my_tqdm(file_paths):
        in_path = Path(in_path)
        assert check_exist(in_path)

        # output to a tmp file first
        tmp_out_path = output_dir / f"{in_path.stem}.tmp.txt"
        if tmp_out_path.exists():
            tmp_out_path.unlink()

        # will skip processed inputs
        out_path = output_dir / f"{in_path.stem}.txt"
        if out_path.exists():
            continue

        waveform = vad_model.silero_read_audio(in_path.as_posix(), sampling_rate=SAMPLE_RATE)
        speech_timestamps: List[dict] = vad_model(waveform, sampling_rate=SAMPLE_RATE)

        if len(speech_timestamps) == 0:
            logger.info(f"{in_path} has none speech parts.")
            # create an empty file as placeholder
            with open(tmp_out_path, mode="w"):
                pass
        else:
            with open(tmp_out_path, mode="w") as fp:
                for activity in speech_timestamps:
                    fp.write(f"{activity['start']} {activity['end']}\n")
        # rename
        tmp_out_path.replace(out_path)


def main():
    args = parse_args()
    logger.info(args)

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)
    if args.use_tgt:
        all_inputs = [pair[1] for pair in all_pairs]
    else:
        all_inputs = [pair[0] for pair in all_pairs]
    all_inputs = sorted(list(set(all_inputs)))  # unique, sorted list

    start, end = get_shard_range(len(all_inputs), nshard=args.n_shard, rank=args.rank)
    all_inputs = all_inputs[start:end]
    logger.info(f"{len(all_inputs)} total || Example of inputs: {all_inputs[:3]}")

    output_dir = Path(args.out_dir) / args.lang
    output_dir.mkdir(parents=True, exist_ok=True)

    vad(
        vad_version=args.vad_version,
        file_paths=all_inputs,
        output_dir=output_dir,
        cache_dir=args.cache_dir
    )
    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

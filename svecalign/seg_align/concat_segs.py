# This file is based on https://github.com/thompsonb/vecalign/blob/master/overlap.py
# The original copyright notice:
"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# We have modified the original script to support speech segments, instead of text sentences.

import argparse
from pathlib import Path
from typing import List, Tuple, Union

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
        "--lang", type=str, required=True,
        help="the language code."
    )
    parser.add_argument(
        "--use_tgt", action="store_true", default=False,
        help="whether to use the target side."
    )
    parser.add_argument(
        "--num_overlaps", type=int, default=5,
        help="Maximum number of allowed overlaps."
    )
    parser.add_argument(
        "--max_dur", type=float, default=20.0,
        help="Maximum time each concatenated segment can have. (in seconds)"
    )
    return parser.parse_args()


def get_overlaps(
        in_path: Union[str, Path],
        num_overlaps: int,
        max_frames: int,
) -> List[str]:
    assert num_overlaps > 0, num_overlaps

    segs = read_segments(in_path)

    overlaps = []

    for i, (start, end) in enumerate(segs):
        # a single segment could also exceed max frames
        if end - start > max_frames:
            continue

        overlaps.append(f"{start} {end}")
        # from i+1 ~ i+num_overlaps-1
        # num_overlaps=[2, num_overlaps]
        for j in range(1, num_overlaps):
            # out of bound
            if i + j >= len(segs):
                break
            # too long
            if segs[i + j][1] - start > max_frames:
                break

            overlaps.append(f"{start} {segs[i + j][1]}")
    return overlaps


def overlap(
        in_path: Union[str, Path],
        out_path: Path,
        num_overlaps: int,
        min_dur: float = 0.0,
        max_dur: float = 30.0,
        sample_rate: int = SAMPLE_RATE,
):
    """
    Overlap for at most `num_overlaps` consecutive segments.
    """
    min_frames = int(min_dur * sample_rate)  # currently unused
    max_frames = int(max_dur * sample_rate)

    overlaps = get_overlaps(
        in_path, num_overlaps,
        max_frames=max_frames,
    )

    if len(overlaps) == 0:
        logger.warning(f"encountered 0 line from {in_path}")

    # out
    overlaps = sorted(overlaps)  # for reproducibility
    logger.debug(f"Got {len(overlaps)} segments")

    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    with open(out_path, mode="w") as fp:
        for overlap_str in overlaps:
            fp.write(f"{overlap_str}\n")


def main():
    args = parse_args()
    logger.info(args)

    num_overlaps = args.num_overlaps
    max_dur = args.max_dur
    logger.info(f"Will concatenate {num_overlaps} consecutive segments. Max dur is {max_dur} sec.")

    all_pairs: List[Tuple[str, str]] = read_metadata(args.metadata)
    if args.use_tgt:
        all_inputs = [pair[1] for pair in all_pairs]
    else:
        all_inputs = [pair[0] for pair in all_pairs]
    all_inputs = sorted(list(set(all_inputs)))  # unique, sorted list

    lang = args.lang
    seg_dir = Path(args.seg_dir) / lang

    out_dir = Path(args.out_dir) / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in my_tqdm(all_inputs):
        audio_stem = Path(audio_path).stem

        overlap(
            in_path=seg_dir / f"{audio_stem}.txt",
            out_path=out_dir / f"{audio_stem}.txt",
            num_overlaps=num_overlaps,
            max_dur=max_dur,
        )

    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

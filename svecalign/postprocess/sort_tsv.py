import argparse
from pathlib import Path

import svecalign.utils.file_utils as file_utils
from svecalign.utils.log_utils import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_tsv", type=str, required=True,
        help="input tsv"
    )
    parser.add_argument(
        "--out_tsv", type=str, required=True,
        help="output tsv"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)

    out_tsv_path = Path(args.out_tsv)
    assert not out_tsv_path.exists(), f"Output file {out_tsv_path} exists!"

    out_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    logger.info(f"Read input...")
    with file_utils.open(args.in_tsv) as fp:
        for line in fp:
            score, _, _ = line.strip().split("\t")
            data.append(
                (float(score), line.strip())
            )
    data.sort(key=lambda x: -x[0])

    logger.info(f"Output to {out_tsv_path}...")
    with file_utils.open(out_tsv_path, mode="w") as fp:
        for _, line in data:
            fp.write(line + "\n")
    logger.info(f"Finished!")


if __name__ == '__main__':
    main()

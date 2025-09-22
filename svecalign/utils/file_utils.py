import builtins
import gzip
import lzma
from ast import literal_eval
from pathlib import Path
from typing import Union, List, Optional, IO, Tuple

from svecalign.utils.log_utils import logging

logger = logging.getLogger(__name__)


def read_lines(path: Union[str, Path]) -> List[str]:
    res = []
    with open(path) as fp:
        for line in fp:
            res.append(line.strip())
    return res


def read_metadata(path: Union[str, Path]) -> List[Tuple[str, str]]:
    return list(map(lambda x: tuple(x.split("\t")), read_lines(path)))


def check_exist(path: Union[Path, str]) -> bool:
    """
    A helper function to deal with paths in string/Path. Will log if the file does not exist.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        logger.warning(f"{path} does not exist.")
        return False

    return True


def delete_if_exist(path: Union[Path, str], verbose: bool = False):
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        if verbose:
            logger.warning(f"{path} exists. Remove.")
        path.unlink()


# copied from stopes.core.utils
def open(
        filename: Union[Path, str],
        mode: str = "rt",
        encoding: Optional[str] = "utf-8",
) -> IO:
    if len(mode) == 1:
        mode += "t"
    if "b" in mode:
        encoding = None
    filename = Path(filename)
    if filename.suffix == ".gz":
        return gzip.open(filename, encoding=encoding, mode=mode)
    elif filename.suffix == ".xz":
        return lzma.open(filename, encoding=encoding, mode=mode)
    else:
        return builtins.open(filename, encoding=encoding, mode=mode)


def read_segments(path: Union[str, Path]) -> List[Tuple[int, int]]:
    """
    Read a segment file and process it into a list of (start, end) timestamps.
    """
    with open(path) as fp:
        res = []
        for line in fp:
            line = line.strip().split(" ")
            assert len(line) == 2, line
            start, end = int(line[0]), int(line[1])
            res.append((start, end))
    return res


def read_alignments(fin) -> List[Tuple[List[int], List[int]]]:
    alignments = []
    with open(fin, 'rt', encoding="utf-8") as infile:
        for line in infile:
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            if len(fields) < 2:
                raise Exception('Got line "%s", which does not have at least two ":" separated fields' % line.strip())
            try:
                src = literal_eval(fields[0])
                tgt = literal_eval(fields[1])
            except:
                raise Exception('Failed to parse line "%s"' % line.strip())
            alignments.append((src, tgt))

    # I know bluealign files have a few entries missing,
    #   but I don't fix them in order to be consistent previous reported scores
    return alignments


def read_alignments_with_score(fin) -> List[Tuple[List[int], List[int], float]]:
    """
    Mostly identical to above. Also return the scores.
    """
    alignments = []
    with open(fin, 'rt', encoding="utf-8") as infile:
        for line in infile:
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            assert len(fields) == 3, \
                'Got line "%s", which does not have at least two ":" separated fields' % line.strip()
            try:
                src = literal_eval(fields[0])
                tgt = literal_eval(fields[1])
                score = float(fields[2])
            except:
                raise Exception('Failed to parse line "%s"' % line.strip())
            alignments.append((src, tgt, score))

    return alignments


def write_alignment(alignments: List[Tuple[List[int], List[int]]], path: Union[Path, str]) -> None:
    """
    Write alignments in the vecalign way.
    """
    with open(path, mode="w") as fp:
        for src_segs, tgt_segs in alignments:
            fp.write(f"{src_segs}:{tgt_segs}\n")


def alignments_to_timestamps(
        align: Union[str, Path, list],
        src_segs: List[Tuple[int, int]],
        tgt_segs: List[Tuple[int, int]],
        ignore_empty: bool = True
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """
    The input `align` contains indexes only. `src_segs` and `tgt_segs` contains the mapping
    between indexes and timestamps.
    Return the timestamps for each alignment.
    """
    src_aligns = []
    tgt_aligns = []

    if isinstance(align, str) or isinstance(align, Path):
        alignments = read_alignments(align)
    elif isinstance(align, list):
        alignments = align
    else:
        raise TypeError(f"{align} type is unexpected. {type(align)}")

    for src, tgt in alignments:
        if not src or not tgt:
            # those failed to be aligned
            if ignore_empty:
                continue
            else:
                raise Exception(f"Got empty alignments!")

        src_aligns.append(
            (src_segs[src[0]][0], src_segs[src[-1]][1])
        )
        tgt_aligns.append(
            (tgt_segs[tgt[0]][0], tgt_segs[tgt[-1]][1])
        )
    assert len(src_aligns) == len(tgt_aligns)
    return src_aligns, tgt_aligns, len(src_aligns)

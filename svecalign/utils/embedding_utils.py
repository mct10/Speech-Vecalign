# Some functions are copied and adapted from https://github.com/thompsonb/vecalign/blob/master/dp_utils.py
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
from typing import Tuple, List, Optional, Set

import numpy as np

from svecalign.utils.log_utils import logging

EMBED_DIM = 1024
PAD_LABEL = "PAD"

logger = logging.getLogger(__name__)


def preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        logger.warning(f"Encountered empty line.")
        line = '[BLANK_LINE]'
    return line


def load_stopes_embeddings(path: str, mode: str = "mmap") -> np.ndarray:
    """The returned embedding is fp32."""
    from stopes.utils.embedding_utils import Embedding  # noqa
    e = Embedding(path)
    with e.open_for_read(mode) as _e:
        embed = _e.astype(np.float32)
    return embed


def load_np_embeddings(embed_file: str, fp16_embed: bool) -> np.ndarray:
    """The returned embedding is fp32."""
    if fp16_embed:
        line_embeddings = np.fromfile(embed_file, dtype=np.float16, count=-1).astype(np.float32)
    else:
        line_embeddings = np.fromfile(embed_file, dtype=np.float32, count=-1)

    return line_embeddings


def load_sent_embeddings(
        embed_file: str,
        use_stopes: bool = False,
        fp16_embed: bool = False,
        stopes_mode: str = "mmap"
) -> np.ndarray:
    """
    It should return only fp32 embeddings.
    """
    if use_stopes:
        line_embeddings = load_stopes_embeddings(embed_file, mode=stopes_mode)
    else:
        # Read sentence embeddings
        line_embeddings = load_np_embeddings(embed_file, fp16_embed)
        if line_embeddings.size == 0:
            raise Exception('Got empty embedding file')

        line_embeddings.resize(line_embeddings.shape[0] // EMBED_DIM, EMBED_DIM)
    assert line_embeddings.dtype == np.float32, embed_file
    return line_embeddings


def read_in_embeddings(
        text_file: str, embed_file: str,
        use_stopes: bool = False,
        fp16_embed: bool = False
) -> Tuple[dict, np.ndarray]:
    """
    :param fp16_embed: whether the embedding file is stored with fp16; only effective for numpy ver.
    :param use_stopes: whether the embedding file is obtained by stopes
    :param text_file: candidate sentences
    :param embed_file: corresponding embeddings
    :return: a mapping from candidate sentence to embedding index,
        and a numpy array of the embeddings for each line of the candidates. It must be in fp32.
    """
    # Read candidate sentences -> line number
    sent2line = dict()
    with open(text_file, 'rt', encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if line.strip() in sent2line:
                continue  # allow duplicate lines. we can assume their embeddings are same
                # raise Exception('got multiple embeddings for the same line')
            sent2line[line.strip()] = i

    line_embeddings = load_sent_embeddings(embed_file, use_stopes, fp16_embed)

    return sent2line, line_embeddings


def make_overlap(
        lines: List[str],
        num_overlaps: int,
        start_id: int,
        ignore_indices: Optional[Set[Tuple[int, int]]] = None,
        comb: str = ' ',
        overlap_segments: bool = False
) -> List[str]:
    """
    overlap_segments: whether working on speech segments.
    """
    res = []
    for n_over in range(num_overlaps):
        j = start_id + n_over
        if j >= len(lines):
            break

        # we will stop overlapping from (start_id, j)
        if ignore_indices and (start_id, j) in ignore_indices:
            res.extend([PAD_LABEL] * (min(len(lines), start_id + num_overlaps) - j))
            break

        if overlap_segments:
            res.append(f"{lines[start_id].split()[0]} {lines[j].split()[1]}")
        else:
            res.append(comb.join(lines[start_id:j + 1]))
    return res


def make_doc_embedding(
        sent2id: dict, line_embeddings: np.ndarray, lines: List[str], max_overlaps: int,
        ignore_indices: Optional[Set[Tuple[int, int]]] = None,
        overlap_segments: bool = False
) -> np.ndarray:
    """
    :param ignore_indices: for some overlaps, we do not align them, so we do not load their embeddings and use 0 instead
    :param sent2id: sentence to index in embeddings
    :param line_embeddings: precomputed embeddings for lines (and overlaps of lines)
    :param lines: sentences in input document to embed
    :param max_overlaps: max of this number of consecutive sentences can form a candidate sentence.
        Must have a corresponding embedding for each candidate.
    :param overlap_segments: if True, treat each line as starting frame and end frame. The overlapped line contains
        only the starting and end frame ids.
    :return: a matrix, containing embeddings for all overlaps. shape = (max_overlaps, #lines, embed_dim).
        matrix[j,i,:] means overlap=j+1 (j=0 or j+1=1 means single sentence),
            embeddings of sentences [i-j, i], j sentences END at pos i.
        More intuitively, it is like this:
        [sent 0,   sent 1,   sent 2,   sent 3,   sent 4,   ..., sent n    ]
        [PAD,      sent 0~1, sent 1~2, sent 2~3, sent 3~4, ..., sent n-1~n]
        [PAD,      PAD,      sent 0~2, sent 1~3, sent 2~4, ..., sent n-2~n]
    """

    lines = [preprocess_line(line) for line in lines]

    embed_dim = line_embeddings.shape[1]

    n_miss = n_match = 0

    candidate_vectors = np.zeros((max_overlaps, len(lines), embed_dim), dtype=np.float32)
    for i in range(len(lines)):  # overlap starting from each sentence
        for j, out_line in enumerate(
                make_overlap(
                    lines, max_overlaps,
                    start_id=i,
                    ignore_indices=ignore_indices, overlap_segments=overlap_segments
                )
        ):
            try:
                if out_line == PAD_LABEL:
                    line_id = None
                else:
                    line_id = sent2id[out_line]
            except KeyError:
                line_id = None

            if line_id is not None:
                vec = line_embeddings[line_id]
                if np.any(np.isnan(vec)):
                    n_miss += 1
                    logger.error(f"loaded a vector with nan value at {line_id} with overlap {out_line}. "
                                 f"Please double check. "
                                 f"Will reset to zero.")
                    with np.printoptions(threshold=embed_dim):
                        logger.error(vec)
                    vec = np.zeros(shape=(embed_dim,), dtype=np.float32)
                else:
                    n_match += 1
            else:
                # will use 0 embeddings to prevent some segments from being aligned
                vec = np.zeros(shape=(embed_dim,), dtype=np.float32)
                n_miss += 1

            # in a diagonal way:
            # 1st sent starts at pos: (0, 0), (1, 1), (2, 2) ...
            # 2nd sent starts at pos: (0, 1), (1, 2), (2, 3), ...
            candidate_vectors[j, i + j, :] = vec
    logger.debug(f"Match: {n_match} || Miss: {n_miss}")
    return candidate_vectors

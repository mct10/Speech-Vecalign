# This file is based on https://github.com/thompsonb/vecalign/blob/master/vecalign.py
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

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union, Set

from svecalign.vecalign.dp_utils import vecalign
from svecalign.vecalign.score import score_multiple, log_final_scores
from svecalign.utils.embedding_utils import read_in_embeddings, make_doc_embedding
from svecalign.utils.file_utils import read_alignments
from svecalign.utils.log_utils import logging

logger = logging.getLogger("vecalign")
logger.propagate = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--src", type=str, dest="src", required=True,
        help='Source file.'
    )
    parser.add_argument(
        '-t', '--tgt', type=str, dest="tgt", required=True,
        help='Target file.'
    )
    parser.add_argument(
        '--src_embed', type=str, nargs=2, required=True,
        help='Source embeddings.'
             'Requires two arguments: first is a text file, second is a binary embeddings file. '
    )
    parser.add_argument(
        "--src_stopes", action="store_true", default=False,
        help="Whether the source embedding should be loaded by stopes."
    )
    parser.add_argument(
        "--src_fp16", action="store_true", default=False,
        help="whether the source embedding is stored with fp16. Used for numpy embeddings, e.g., SONAR."
    )
    parser.add_argument(
        '--tgt_embed', type=str, nargs=2, required=True,
        help='Target embeddings.'
             'Requires two arguments: first is a text file, second is a binary embeddings file. '
    )
    parser.add_argument(
        "--tgt_stopes", action="store_true", default=False,
        help="Whether the target embedding should be loaded by stopes."
    )
    parser.add_argument(
        "--tgt_fp16", action="store_true", default=False,
        help="whether the target embedding is stored with fp16. Used for numpy embeddings, e.g., SONAR."
    )
    parser.add_argument(
        '-a', '--alignment_max_size', dest="alignment_max_size", type=int, default=10,
        help='Searches for alignments up to size N-M, where N+M <= this value.'
             'Note that the the embeddings must support the requested number of overlaps'
    )
    # without flag: one_to_many==default;
    # with flag but no argument: one_to_many==const;
    # with flag and argument: one_to_many==argument
    # in speech-text case, it is likely to have multiple segments aligned with one sentence
    parser.add_argument(
        '--many_to_one', type=int, nargs='?', default=None, const=50,
        help='Perform many to one (e.g. 1:1, 2:1, ... M:1) alignment.'
             'Argument specifies M but will default to 50 if flag is set but no argument is provided. '
             'Overrides --alignment_max_size (-a).'
    )
    parser.add_argument(
        '-d', '--del_percentile_frac', type=float, default=0.2,
        help='Deletion penalty is set to this percentile (as a fraction) of the cost matrix distribution. '
             'Should be between 0 and 1.'
    )
    parser.add_argument(
        '--search_buffer_size', type=int, default=5,
        help='Width (one side) of search buffer. '
             'Larger values makes search more likely to recover from errors but increases runtime.'
    )
    parser.add_argument(
        '--max_size_full_dp', type=int, default=300,
        help='Maximum size N for which is is acceptable to run full N^2 dynamic programming.'
    )
    parser.add_argument(
        '--costs_sample_size', type=int, default=20000,
        help='Sample size to estimate costs distribution, used to set deletion penalty in conjunction with deletion_percentile.'
    )

    parser.add_argument(
        '--num_samps_for_norm', type=int, default=100,
        help='Number of samples used for normalizing embeddings'
    )
    parser.add_argument(
        "--overlap_segments", default=False, action="store_true",
        help="defines the overlap method. If True (for speech), treat each line as containing starting and end frames."
    )
    parser.add_argument(
        "--src_ignore_indices", default=None, type=str,
        help="if provided, will not load the embeddings starting from these indices."
    )
    parser.add_argument(
        "--tgt_ignore_indices", default=None, type=str,
        help="if provided, will not load the embeddings starting from these indices."
    )
    parser.add_argument(
        '-g', '--gold_alignment',
        dest="gold_alignment",
        type=str,
        required=False,
        default=None,
        help='preprocessed target file to align'
    )
    parser.add_argument(
        '--print_aligned_text', action='store_true',
        help='Print aligned text in addition to alignments, for debugging/tuning.'
    )
    parser.add_argument(
        "--save_to_file", type=str, default=None,
        help="If not None, write to the provided file."
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true",
        help='sets consle to logging.DEBUG instead of logging.WARN'
    )
    parser.add_argument(
        '--debug_save_stack', type=str, default=None,
        help='Write stack to pickle file for debug purposes'
    )
    parser.add_argument(
        "--print_results", default=False, action="store_true",
        help="whether to print results at all."
    )
    args = parser.parse_args()
    return args


def make_alignment_types(max_alignment_size: int):
    # return list of all (n,m) where n+m <= max_alignment_size
    # does not include deletions, i.e. (1, 0) or (0, 1)
    alignment_types = []
    for x in range(1, max_alignment_size):
        for y in range(1, max_alignment_size):
            if x + y <= max_alignment_size:
                alignment_types.append((x, y))
    return alignment_types


def make_many_to_one_alignment_types(max_alignment_size: int):
    # return list of all (m, 1) where m <= max_alignment_size
    # does not include deletions, i.e. (1, 0) or (0, 1)
    alignment_types = []
    for m in range(1, max_alignment_size + 1):
        alignment_types.append((m, 1))
    return alignment_types


def print_alignments(alignments, scores=None, src_lines=None, tgt_lines=None, ofile=sys.stdout):
    if scores is None:
        scores = [None for _ in alignments]
    for (x, y), s in zip(alignments, scores):
        if s is None:
            print('%s:%s' % (x, y), file=ofile)
        else:
            print('%s:%s:%.6f' % (x, y, s), file=ofile)
        if src_lines is not None and tgt_lines is not None:
            print(' ' * 40, 'SRC: ', ' '.join([src_lines[i].replace('\n', ' ').strip() for i in x]), file=ofile)
            print(' ' * 40, 'TGT: ', ' '.join([tgt_lines[i].replace('\n', ' ').strip() for i in y]), file=ofile)


def load_ignore_index_file(path: Union[str, Path]) -> Set[Tuple[int, int]]:
    with open(path) as fp:
        res = set()
        for line in fp:
            i, j = line.strip().split(" ")
            item = (int(i), int(j))
            assert item not in res, f"{path}, {item}"
            res.add(item)
    return res


def align(
        src: str,
        tgt: str,
        src_embed: List[str],
        src_stopes: bool,
        tgt_stopes: bool,
        tgt_embed: List[str],
        alignment_max_size: int,
        many_to_one: Optional[int],
        search_buffer_size: int,
        del_percentile_frac: float,
        max_size_full_dp: int,
        costs_sample_size: int,
        num_samps_for_norm: int,
        overlap_segments: bool,
        print_aligned_text: bool,
        src_fp16: bool = False,
        tgt_fp16: bool = False,
        src_ignore_indices: Optional[Union[str, Path]] = None,
        tgt_ignore_indices: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        debug_save_stack: Optional[str] = None,
        gold_alignment: Optional[str] = None,
        print_results: bool = False,
        save_aligned_text_to_file: Optional[str] = None
):
    """
    This is adapted to support aligning a single pair of documents.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    if alignment_max_size < 2:
        logger.warning('Alignment_max_size < 2. Increasing to 2 so that 1-1 alignments will be considered')
        alignment_max_size = 2

    src_max_alignment_size = many_to_one if many_to_one is not None else alignment_max_size - 1
    tgt_max_alignment_size = 1 if many_to_one is not None else alignment_max_size - 1

    if many_to_one is not None:
        final_alignment_types = make_many_to_one_alignment_types(many_to_one)
    else:
        final_alignment_types = make_alignment_types(alignment_max_size)
    logger.debug('Considering alignment types %s', final_alignment_types)

    width_over2 = math.ceil(max(src_max_alignment_size, tgt_max_alignment_size) / 2.0) + search_buffer_size

    src_sent_to_id, src_embeddings = read_in_embeddings(src_embed[0], src_embed[1], src_stopes, src_fp16)
    tgt_sent_to_id, tgt_embeddings = read_in_embeddings(tgt_embed[0], tgt_embed[1], tgt_stopes, tgt_fp16)

    logger.info(f'Aligning src={src} to tgt={tgt}')

    src_lines = open(src, 'rt', encoding="utf-8").readlines()
    src_vectors = make_doc_embedding(
        src_sent_to_id, src_embeddings, src_lines, src_max_alignment_size,
        ignore_indices=load_ignore_index_file(src_ignore_indices) if src_ignore_indices else None,
        overlap_segments=overlap_segments
    )

    tgt_lines = open(tgt, 'rt', encoding="utf-8").readlines()
    tgt_vectors = make_doc_embedding(
        tgt_sent_to_id, tgt_embeddings, tgt_lines, tgt_max_alignment_size,
        ignore_indices=load_ignore_index_file(tgt_ignore_indices) if tgt_ignore_indices else None,
        overlap_segments=overlap_segments
    )

    stack = vecalign(
        vecs0=src_vectors,
        vecs1=tgt_vectors,
        final_alignment_types=final_alignment_types,
        del_percentile_frac=del_percentile_frac,
        width_over2=width_over2,
        max_size_full_dp=max_size_full_dp,
        costs_sample_size=costs_sample_size,
        num_samps_for_norm=num_samps_for_norm
    )

    # write final alignments to stdout or a file
    if print_results:
        detail_fp = open(save_aligned_text_to_file, mode="w") if save_aligned_text_to_file else sys.stdout
        print_alignments(
            stack[0]['final_alignments'], scores=stack[0]['alignment_scores'],
            src_lines=src_lines if print_aligned_text else None,
            tgt_lines=tgt_lines if print_aligned_text else None,
            ofile=detail_fp
        )
        if save_aligned_text_to_file:
            detail_fp.close()

    if debug_save_stack:
        pickle.dump(stack, open(debug_save_stack, mode="wb"))

    if gold_alignment is not None:
        gold_list = read_alignments(gold_alignment)
        res = score_multiple(gold_list=[gold_list], test_list=[stack[0]['final_alignments']])
        log_final_scores(res)


if __name__ == '__main__':
    _args = parse_args()
    align(**vars(_args))

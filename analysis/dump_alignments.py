# This script merge segments aligned by speech_vecalign.py,
# and save to new files.
import argparse
import dataclasses
import logging
import math
from pathlib import Path
from typing import List, Tuple, Union, Optional

from svecalign.utils.audio_utils import load_waveform
from svecalign.utils.embed_model_utils import save_segment_audio_and_tsv
from svecalign.utils.file_utils import read_segments, alignments_to_timestamps, \
    read_alignments, \
    read_alignments_with_score
from svecalign.utils.mp_utils import get_shard_range

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--align_path", type=str, required=True,
    )
    parser.add_argument(
        "--src_segs", type=str, required=True,
    )
    parser.add_argument(
        "--src_wav", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_segs", type=str, required=True,
    )
    parser.add_argument(
        "--tgt_wav", type=str, required=True,
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
    )
    # for Whisper setups
    parser.add_argument(
        "--asr",
        default=False,
        action="store_true",
        help="if true, will run Whisper to generate the transcriptions and write HTMLs."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default=None,
        help="whisper lang code"
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default=None,
        help="whisper lang code"
    )
    parser.add_argument(
        "--whisper_size",
        type=str,
        default="medium",
        help="which whisper to use. medium is usually good enough."
    )
    parser.add_argument(
        "--whisper_root",
        type=str,
        default="./",
        help="where to save whisper model"
    )
    return parser.parse_args()


@dataclasses.dataclass
class Alignment:
    score: float

    src_seg_ids: List[int]
    tgt_seg_ids: List[int]

    src_wav_path: str
    tgt_wav_path: str

    src_transcript: str
    tgt_transcript: str


def pack_segments(
        src_wav_paths: List[str],
        src_transcripts: List[str],
        tgt_wav_paths: List[str],
        tgt_transcripts: List[str],
        alignments: Union[List[Tuple[List[int], List[int], float]], List[Tuple[List[int], List[int]]]]
) -> List[Alignment]:
    assert len(src_wav_paths) == len(tgt_wav_paths)
    assert len(src_transcripts) == len(tgt_transcripts)

    src_id = tgt_id = 0  # for audio id
    res = []
    for item in alignments:
        if len(item) == 2:
            src_seg, tgt_seg, score = list(item) + [0.0]
        else:
            src_seg, tgt_seg, score = item
        assert src_seg or tgt_seg
        # tgt del
        if not tgt_seg:
            res.append(
                Alignment(
                    score,
                    src_seg_ids=src_seg, tgt_seg_ids=tgt_seg,
                    src_wav_path="", tgt_wav_path="",
                    src_transcript="", tgt_transcript=""
                )
            )
        # src del
        elif not src_seg:
            res.append(
                Alignment(
                    score,
                    src_seg_ids=src_seg, tgt_seg_ids=tgt_seg,
                    src_wav_path="", tgt_wav_path="",
                    src_transcript="", tgt_transcript=""
                )
            )
        # have both
        else:
            res.append(
                Alignment(
                    score,
                    src_seg_ids=src_seg, tgt_seg_ids=tgt_seg,
                    src_wav_path=src_wav_paths[src_id], tgt_wav_path=tgt_wav_paths[tgt_id],
                    src_transcript=src_transcripts[src_id], tgt_transcript=tgt_transcripts[tgt_id]
                )
            )
            src_id += 1
            tgt_id += 1
    return res


def asr(
        whisper_model,
        tsv_path: Path,
        lang: str
) -> Tuple[List[str], List[str]]:
    transc = []
    wavs = []
    with open(tsv_path) as fp:
        # /path/to/output/wavs/ -> wavs/
        base_dir = Path(fp.readline().strip())
        rel_dir = Path(base_dir.stem)
        for line in fp:
            sub_path, _ = line.strip().split("\t")
            wav_path = (base_dir / sub_path).as_posix()
            audio = load_waveform(wav_path)
            txt = whisper_model.transcribe(
                audio,
                language=lang,
                fp16=True
            )["text"]
            transc.append(txt)
            wavs.append((rel_dir / sub_path).as_posix())
    return wavs, transc


def write_single_html(
        align_meta: List[Alignment],
        out_path: Path
):
    ctx = '<table>\n'

    # header
    ctx += "\t<tr>\n" \
           "\t\t<th>Score</th>\n" \
           "\t\t<th>Src Segs</th>\n" \
           "\t\t<th>Src Txt</th>\n" \
           "\t\t<th>Src Audio</th>\n" \
           "\t\t<th>Tgt Segs</th>\n" \
           "\t\t<th>Tgt Txt</th>\n" \
           "\t\t<th>Tgt Audio</th>\n" \
           "\t</tr>\n"

    for collect in align_meta:
        ctx += f"\t<tr>\n" \
               f"\t\t<td>{collect.score}</td>\n" \
               f"\t\t<td>{collect.src_seg_ids}</td>\n" \
               f"\t\t<td>{collect.src_transcript}</td>\n" \
               f"\t\t<td><audio controls><source src=\"{collect.src_wav_path}\" type=\"audio/wav\"></audio></td>\n" \
               f"\t\t<td>{collect.tgt_seg_ids}</td>\n" \
               f"\t\t<td>{collect.tgt_transcript}</td>\n" \
               f"\t\t<td><audio controls><source src=\"{collect.tgt_wav_path}\" type=\"audio/wav\"></audio></td>\n" \
               f"\t</tr>\n"
    ctx += '</table>'
    with open(out_path, "w") as fp:
        fp.write(ctx)
    print(f"Wrote HTML to {out_path}")


def dump(
        src_wav_path: str,
        tgt_wav_path: str,
        src_seg_path: str,
        tgt_seg_path: str,
        align_path: str,
        out_dir: str,
        apply_asr: bool = False,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        whisper_size: str = "medium",
        whisper_root: str = "./"
):
    src_segs = read_segments(src_seg_path)
    tgt_segs = read_segments(tgt_seg_path)
    src_aligns, tgt_aligns, n_aligns = alignments_to_timestamps(align_path, src_segs, tgt_segs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # src
    logger.info(f"Save audios for src")
    save_segment_audio_and_tsv(
        out_dir=out_dir,
        wave_path=src_wav_path,
        segments=src_aligns,
        wav_dir_name="src_wavs",
        tsv_file_name="src.tsv",
        ext="ogg"
    )

    # target
    logger.info(f"Save audios for tgt")
    save_segment_audio_and_tsv(
        out_dir=out_dir,
        wave_path=tgt_wav_path,
        segments=tgt_aligns,
        wav_dir_name="tgt_wavs",
        tsv_file_name="tgt.tsv",
        ext="ogg"
    )

    if not apply_asr:
        return

    import whisper  # noqa

    whisper_model = whisper.load_model(
        whisper_size,
        device="cuda:0",
        download_root=whisper_root
    ).eval()

    logger.info(f"ASR src")
    src_wav_paths, src_trans = asr(whisper_model, out_dir / "src.tsv", src_lang)
    logger.info(f"ASR tgt")
    tgt_wav_paths, tgt_trans = asr(whisper_model, out_dir / "tgt.tsv", tgt_lang)

    try:
        alignments = read_alignments_with_score(align_path)
    except AssertionError:
        alignments = read_alignments(align_path)
    align_meta = pack_segments(src_wav_paths, src_trans, tgt_wav_paths, tgt_trans, alignments)

    n_shards = math.ceil(len(align_meta) / 100)
    for i in range(n_shards):
        start, end = get_shard_range(len(align_meta), n_shards, i)
        write_single_html(align_meta[start:end], out_dir / f"main_{i}.html")
    logger.info(f"Finished!")


def main():
    args = parse_args()
    logger.info(args)

    dump(
        src_wav_path=args.src_wav, tgt_wav_path=args.tgt_wav,
        src_seg_path=args.src_segs, tgt_seg_path=args.tgt_segs,
        align_path=args.align_path,
        out_dir=args.out_dir,
        apply_asr=args.asr,
        src_lang=args.src_lang, tgt_lang=args.tgt_lang,
        whisper_size=args.whisper_size, whisper_root=args.whisper_root
    )


if __name__ == '__main__':
    main()

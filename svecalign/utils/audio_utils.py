from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import soundfile as sf
import torch

from svecalign.utils.log_utils import logging

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def load_waveform(
        path: Union[str, Path], start: int = 0, end: Optional[int] = None, sr: int = SAMPLE_RATE
) -> np.ndarray:
    if isinstance(path, Path):
        path = path.as_posix()

    waveform, _sr = sf.read(path, dtype="float32", start=start, stop=end)
    assert _sr == sr, f"Expected sample rate {sr} but got {_sr}"
    return waveform


def save_waveform(
        waveform: np.ndarray, target: Union[str, Path],
        sr: int = SAMPLE_RATE, overwrite_wav: bool = False
):
    if isinstance(target, str):
        target = Path(target)

    if target.exists() and not overwrite_wav:
        return
    sf.write(target.as_posix(), waveform, samplerate=sr)


def find_untranslated_segs(
        segments_1: List[Tuple[int, int]],
        segments_2: List[Tuple[int, int]],
        audio_1_path: Union[str, Path],
        audio_2_path: Union[str, Path],
        max_frame_diff: int,
        fbank_dist_thres: float
) -> List[Tuple[int, int]]:
    """
    Given source and target segments, find the identical untranslated ones.
    Return `indices` of the source and target segments.
    """
    if len(segments_1) == 0 or len(segments_2) == 0:
        return []

    # convert to Segment class
    segments_1 = [
        Segment(seg[0], seg[1], audio_1_path)
        for seg in segments_1
    ]
    segments_2 = [
        Segment(seg[0], seg[1], audio_2_path)
        for seg in segments_2
    ]

    # make sure sorted by timestamps such that binary search can be applied
    segs_2_mids = np.array(
        [seg.mid for seg in segments_2],
        dtype=float
    )
    assert np.all(segs_2_mids[:-1] < segs_2_mids[1:]), segs_2_mids

    res = []
    for i, src_seg in enumerate(segments_1):
        # find the closest target segment based on midpoint timestamp
        closest_tgt_id = binary_search(segs_2_mids, src_seg.mid)
        try:
            tgt_seg = segments_2[closest_tgt_id]
        except IndexError as e:
            logger.error(f"{audio_1_path} | {audio_2_path} | {i} | {closest_tgt_id}")
            raise e

        # check duration
        if abs(src_seg.duration - tgt_seg.duration) > max_frame_diff:
            continue

        # check fbank dist
        fbank_dist = compute_fbank_dist(src_seg.fbank(), tgt_seg.fbank())
        if fbank_dist > fbank_dist_thres:
            continue

        # an identical pair detected!
        # add indices
        res.append(
            (i, closest_tgt_id)
        )
    return res


def binary_search(arr: np.ndarray, target: float) -> int:
    _id = np.searchsorted(
        arr, target, sorter=None
    )
    # if already at ends, then got the answer
    if _id == 0:
        return _id
    elif _id == len(arr):
        return _id - 1
    # compare the one to the left (_id - 1) and to the right (_id)
    left = arr[_id - 1]
    right = arr[_id]
    if abs(target - left) > abs(target - right):
        return _id
    else:
        return _id - 1


def compute_fbank_dist(fbank1: torch.Tensor, fbank2: torch.Tensor) -> float:
    # let fbank1 be the shorter one
    if fbank1.shape[0] > fbank2.shape[0]:
        tmp = fbank1
        fbank1 = fbank2
        fbank2 = tmp

    len1 = fbank1.shape[0]
    len2 = fbank2.shape[0]
    if len1 == len2:
        mse = torch.nn.functional.mse_loss(fbank1, fbank2)
        return mse.cpu().item()
    else:
        min_mse = float("inf")
        for i in range(len2 - len1):
            mse = torch.nn.functional.mse_loss(fbank1, fbank2[i:i + len1, :])
            min_mse = min(mse.cpu().item(), min_mse)
        return min_mse


def _get_torchaudio_fbank(
        waveform: np.ndarray, sample_rate, n_bins=80, use_gpu: bool = False
) -> Optional[torch.Tensor]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi

        waveform = torch.from_numpy(waveform)
        if use_gpu:
            waveform = waveform.cuda()

        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
        )
        return features
    except ImportError:
        return None


class Segment:
    """
    A wrapper to process segments more easily.
    """

    def __init__(self, start: int, end: int, path: Union[str, Path]):
        """
        start and end are in frames.
        """
        assert end > start
        self.__start = start
        self.__end = end
        if isinstance(path, str):
            path = Path(path)
        assert path.exists()
        self.__path = path

    @property
    def start(self) -> int:
        return self.__start

    @property
    def end(self) -> int:
        return self.__end

    @property
    def path(self):
        return self.__path

    @property
    def mid(self) -> float:
        return (self.__start + self.__end) / 2

    @property
    def duration(self) -> int:
        return self.__end - self.__start

    @property
    def waveform(self) -> np.ndarray:
        return load_waveform(self.path, start=self.start, end=self.end)

    def fbank(self, use_gpu: bool = False) -> torch.Tensor:
        wav, sr = sf.read(self.path, dtype="float32", always_2d=True,
                          start=self.start, stop=self.end)
        assert sr == SAMPLE_RATE, sr
        wav = wav.T
        return _get_torchaudio_fbank(wav, SAMPLE_RATE, use_gpu=use_gpu)

    def save(self, path: str):
        save_waveform(waveform=self.waveform, target=path)

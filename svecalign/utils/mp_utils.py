from svecalign.utils.log_utils import logging

logger = logging.getLogger(__name__)


# copied from https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/examples/hubert/simple_kmeans/feature_utils.py#L23
def get_shard_range(tot: int, nshard: int, rank: int):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end - start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def start_multi_processes(
        data: list,
        n_proc: int,
        func: callable,
        use_torch_mp: bool = False,
        *args,
        **kwargs
):
    """
    It is required that the first argument of `func` is process id,
        the second argument is the list of data.
    """
    assert n_proc > 0, f"{n_proc} should be larger than 0."

    if len(data) < n_proc:
        n_proc = len(data)

    if n_proc == 1:
        logger.info(f"Single process")
        func(0, data, *args, **kwargs)
    else:
        if use_torch_mp:
            logger.info(f"Torch multi process")
            import torch.multiprocessing as mp
            Process = mp.Process
        else:
            logger.info(f"Python Multi process")
            from multiprocessing import Process as Py_Process
            Process = Py_Process

        processes = []
        for pid in range(n_proc):
            start, end = get_shard_range(len(data), nshard=n_proc, rank=pid)
            processes.append(
                Process(
                    target=func,
                    args=(
                        pid,
                        data[start:end],
                        *args
                    ),
                    kwargs=kwargs
                )
            )
        for p in processes:
            p.start()
        for p in processes:
            p.join()

import numpy as np
from typing import Dict, List

from torch.utils.data.dataloader import DataLoader


def get_batch(dataloader_iter, dataloader):
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
    return batch


class Sampler:
    def __init__(
        self, p: float, dataloaders: List[DataLoader], tasks: List[str], queue_len: int
    ):
        # Sampling Weights
        self.init_p = p

        self.queue_len = queue_len
        self.tasks = tasks
        self.dataloaders = dataloaders
        self.list_of_iters = {k: iter(dataloaders[k]) for k in self.tasks}

    def __iter__(self):
        return self

    def __next__(self):
        current_p = self.init_p

        tasks = np.random.choice(self.tasks, self.queue_len, p=current_p)
        queue = [
            {
                "task": tasks[i],
                "batch": get_batch(
                    self.list_of_iters[tasks[i]], self.dataloaders[tasks[i]]
                ),
            }
            for i in range(self.queue_len)
        ]
        return queue


def UniformBatchSampler(
    dataloaders: List[DataLoader],
    corpus_len: Dict[str, int],
    tasks: List[str],
    temp: float = 1.0,
    queue_len: int = 8,
):
    p = np.array(
        [corpus_len[y] * 1.0 / sum([corpus_len[x] for x in tasks]) for y in tasks]
    )
    p_temp = np.power(p, 1.0 / temp)
    p_temp = p_temp / np.sum(p_temp)

    sampler = Sampler(p_temp, dataloaders, tasks, queue_len)
    samples = iter(sampler)
    return samples

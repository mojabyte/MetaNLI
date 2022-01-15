import torch
from torch.utils.data import Sampler, Dataset
import random
from typing import List, Tuple


def LD2DT(LD):
    return {k: torch.stack([dic[k] for dic in LD]) for k in LD[0]}


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query data from these classes.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        # n_query_way: int,
        n_query: int,
        n_tasks: int,
    ):
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        # self.n_query_way = n_query_way
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.replacement = False

        self.indices_per_label = {}
        if "label" in dataset.data.keys():
            for index, label in enumerate(dataset.data["label"].tolist()):
                if label in self.indices_per_label.keys():
                    self.indices_per_label[label].append(index)
                else:
                    self.indices_per_label[label] = [index]
        else:
            self.indices_per_label[0] = range(len(dataset))
            self.replacement = True

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.indices_per_label[label], (self.n_shot + self.n_query)
                        )
                    )
                    for label in (
                        random.choices(
                            list(self.indices_per_label.keys()), k=self.n_way
                        )
                        if self.replacement
                        else random.sample(self.indices_per_label.keys(), self.n_way)
                    )
                ]
            )

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic data loaders.
        Args:
        input_data: each element is a tuple containing:
            - data as a torch Tensor
            - the label of this data
        Returns:
        list({
            support: {key: Tensor for key in input_data},
            query: {key: Tensor for key in input_data}
        })
        """

        if "label" in input_data[0].keys():
            input_data.sort(key=lambda item: item["label"])

        input_data = LD2DT(input_data)

        def split_tensor(tensor):
            tensor = tensor.reshape(
                (self.n_way, (self.n_shot + self.n_query), *tensor.shape[1:])
            )
            tensor = [
                split.flatten(end_dim=1)
                for split in torch.split(tensor, [self.n_shot, self.n_query], dim=1)
            ]

            return tensor

        data = {k: split_tensor(v) for k, v in input_data.items()}
        data = {
            key: {k: v[j] for k, v in data.items()}
            for j, key in enumerate(["support", "query"])
        }

        return data

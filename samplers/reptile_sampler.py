import random
from typing import List, Tuple

import torch
from torch.utils.data import Sampler, Dataset


def LD2DT(LD):
    return {k: torch.stack([dic[k] for dic in LD]) for k in LD[0]}


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support data from these classes.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_tasks: int,
        reptile_step: int = 3,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support data for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_tasks = n_tasks
        self.reptile_step = reptile_step
        self.replacement = False

        self.indices_per_label = {}

        if "label" in dataset[0].keys():
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
                            self.indices_per_label[label],
                            self.reptile_step * self.n_shot,
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
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            list({key: Tensor for key in input_data}) with length of reptile_step
        """
        if "label" in input_data[0].keys():
            input_data.sort(key=lambda item: item["label"])

        input_data = LD2DT(input_data)

        def chunk_tensor(tensor):
            """
            Function to split the input tensor into a list of support data with
            the length of reptile_step
            Args:
                tensor: input tensor (number of samples) x (data dimension)
            Returns:
                list([
                    Tensor((n_way * n_shot) x (data dimension))
                ]) with the length of reptile_step
            """
            tensor = tensor.reshape(
                (self.n_way, self.reptile_step * self.n_shot, *tensor.shape[1:])
            )

            return [
                item.flatten(end_dim=1)
                for item in torch.chunk(tensor, self.reptile_step, dim=1)
            ]

        data = {k: chunk_tensor(v) for k, v in input_data.items()}
        data = [{k: v[i] for k, v in data.items()} for i in range(self.reptile_step)]

        return data

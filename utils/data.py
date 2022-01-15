import os
import pandas as pd
import pickle5 as pickle

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.data.processors.squad import (
    SquadV1Processor,
    squad_convert_examples_to_features,
)


class CorpusNLI(Dataset):
    def __init__(self, path, model_name="xlm-roberta-base", local_files_only=False):
        self.max_sequence_length = 128

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=False, local_files_only=local_files_only
        )

        self.label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}

        cached_data_file = path + f"_{type(self.tokenizer).__name__}.pickle"

        if os.path.exists(cached_data_file):
            self.data = pickle.load(open(cached_data_file, "rb"))
        else:
            self.data = self.preprocess(path)
            with open(cached_data_file, "wb") as f:
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def preprocess(self, path):
        labels = []
        input_ids = []
        token_type_ids = []
        attention_mask = []
        header = ["premise", "hypothesis", "label"]

        df = pd.read_csv(path, sep="\t", header=None, names=header)

        premise_list = df["premise"].to_list()
        hypothesis_list = df["hypothesis"].to_list()
        label_list = df["label"].to_list()

        # Tokenize input pair sentences
        ids = self.tokenizer(
            premise_list,
            hypothesis_list,
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        input_ids = ids["input_ids"]
        attention_mask = ids["attention_mask"]
        token_type_ids = ids["token_type_ids"]

        labels = torch.tensor([self.label_dict[label] for label in label_list])

        dataset = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }

        return dataset

    def __len__(self):
        return self.data["input_ids"].shape[0]

    def __getitem__(self, id):
        return {
            "input_ids": self.data["input_ids"][id],
            "token_type_ids": self.data["token_type_ids"][id],
            "attention_mask": self.data["attention_mask"][id],
            "label": self.data["label"][id],
        }


class CorpusQA(Dataset):
    def __init__(
        self, path, evaluate, model_name="xlm-roberta-base", local_files_only=False
    ):
        self.doc_stride = 128
        self.max_query_len = 64
        self.max_seq_len = 384

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            use_fast=False,
            local_files_only=local_files_only,
        )

        self.data, self.examples, self.features = self.preprocess(path, evaluate)

    def preprocess(self, file, evaluate=False):
        file = file.split("/")
        filename = file[-1]
        data_dir = "/".join(file[:-1])

        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}".format(type(self.tokenizer).__name__, filename)
        )

        # Init features and dataset from cache if it exists
        if os.path.exists(cached_features_file):
            features_and_dataset = torch.load(cached_features_file)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            processor = SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(data_dir, filename)
            else:
                examples = processor.get_train_examples(data_dir, filename)

            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_len,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_len,
                is_training=not evaluate,
                return_dataset="pt",
                threads=1,
            )

            torch.save(
                {"features": features, "dataset": dataset, "examples": examples},
                cached_features_file,
            )

        return dataset, examples, features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return {
            "input_ids": self.data[id][0],
            "attention_mask": self.data[id][1],
            "token_type_ids": self.data[id][2],
            "answer_start": self.data[id][3],
            "answer_end": self.data[id][4],
        }

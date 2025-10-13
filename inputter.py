#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from utils import norm_string
from easy_tools.utils.io import FileReader


class CPCURDataset(Dataset):
    def __init__(self, file_path: str, dataset_name: str):
        super().__init__()

        assert file_path.endswith("csv")
        self.file_path = file_path
        self.dataset_name = dataset_name.lower()

        data = FileReader.read(file_path)

        self.events = data.event
        self.relations = data.rel
        self.thens = data.then
        self.size = data.shape[0]

        if dataset_name == "atomic":
            self.mbtis = [-1] * self.size
        else:
            self.mbtis = data.mbti

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        current_event = norm_string(
            " ".join(
                f"[{s}]" if s in ["PersonX", "PersonY"] else s
                for s in str(self.events[index]).split()
            )
        )
        current_relation = norm_string("[" + str(self.relations[index]) + "]")
        current_then = norm_string(
            " ".join(
                f"[{s}]" if s in ["PersonX", "PersonY"] else s
                for s in str(self.thens[index]).split()
            )
        )

        current_mbti = self.mbtis[index]
        if self.dataset_name != "atomic":
            current_mbti = norm_string(str(self.mbtis[index]))

        return {
            "events": current_event,
            "relations": current_relation,
            "thens": current_then,
            "mbtis": current_mbti,
        }

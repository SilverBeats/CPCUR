#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from easy_tools.utils.io import FileReader, FileWriter
from easy_tools.utils.trainer import TrainingArguments
from torch.utils.data import DataLoader

from inputter import CPCURDataset
from trainer import CPCURTrainer, evaluate_model
from model import GPT2CSModel, BartCSModel, BertCSModel, CSModel, PureCSModel
from utils import MyMetric

CONFIG_FILE_PATH = "config.yaml"
MODEL_TYPE_TO_CLS = {
    "gpt2": GPT2CSModel,
    "bart": BartCSModel,
    "bert": BertCSModel,
    "none": PureCSModel,
}


def load_config():
    config = FileReader.read(CONFIG_FILE_PATH, return_dict=True)

    if config["ablation_name"] in ["None", "none", None]:
        config["ablation_name"] = "none"

    ablation_name = config["ablation_name"]

    dataset_name = config["dataset_name"].lower()
    model_type = config["model_type"].lower()

    assert dataset_name in ["atomic", "cpcur"]
    assert ablation_name in ["none", "a", "c", "ac"]
    assert model_type in MODEL_TYPE_TO_CLS

    model_config = config["model_config"][model_type]

    if ablation_name == "ac":
        assert model_config["name_or_path"]["a"] is not None
        model_config["name_or_path"] = model_config["name_or_path"]["a"]
    else:
        assert model_config["name_or_path"]["none"] is not None
        model_config["name_or_path"] = model_config["name_or_path"]["none"]

    if ablation_name == "a":
        assert dataset_name == "atomic"
    elif ablation_name in ["c", "ac"]:
        assert dataset_name == "cpcur"

    config["training_args"]["output_dir"] = os.path.join(
        config["training_args"]["output_dir"],
        f"model_type_{model_type}_ablation_name_{ablation_name}_dataset_name_{dataset_name}",
    )
    os.makedirs(config["train_config"]["output_dir"], exist_ok=True)
    return config


def build_model(config: dict) -> CSModel:
    model_type = config["model_type"]
    model_config = config["model_config"][model_type]
    model = MODEL_TYPE_TO_CLS[model_type](**model_config)
    return model


def main():
    config = load_config()
    model = build_model(config)

    dataset_name = config["dataset_name"]
    if config["ablation_name"]:
        training_args = TrainingArguments(**config["training_args"])
        CPCURTrainer(
            dataset_name=dataset_name,
            eval_metric_config=config["evaluate_config"],
            model=model,
            train_file_path=config["file_paths"][dataset_name]["train"],
            config=training_args,
            val_file_path=config["file_paths"][dataset_name]["valid"],
        ).train()
    else:
        results = evaluate_model(
            model=model,
            dataloader=DataLoader(
                dataset=CPCURDataset(
                    config["file_paths"][dataset_name]["valid"], dataset_name
                ),
                batch_size=config["training_args"]["eval_batch_size"],
            ),
            metric_obj=MyMetric(**config["evaluate_config"]),
            dataset_name=dataset_name,
            train_file_path=config["file_paths"][dataset_name]["train"],
            is_pure=True,
        )
        file_path = os.path.join(
            config["training_args"]["output_dir"],
            f"{dataset_name}_metric_result.json",
        )
        FileWriter.dump(results, file_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import List

import torch
from easy_tools.utils.constant import LOGGER
from easy_tools.utils.tools import rm_dir
from easy_tools.utils.trainer import Trainer
from torch.utils.data import DataLoader
from utils import MyMetric, calc_diversity, norm_string
from tqdm import tqdm
from inputter import CPCURDataset


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    metric_obj: MyMetric,
    dataset_name: str,
    train_file_path: str,
    is_pure: bool = False,
):
    model.eval()
    if not is_pure:
        total_loss, total_samples = 0, 0
    gen_data, refs, hyps = [], [], []
    for batch in tqdm(dataloader, desc="evaluating", dynamic_ncols=True):
        if not is_pure:
            loss = model(**batch)["loss"]
            cur_samples = len(gens)
            total_loss += loss.item() * cur_samples
            total_samples += cur_samples

        gens: List[str] = model.generate(**batch)
        gens = [norm_string(s) for s in gens]

        for r, g in zip(batch["thens"], gens):
            metric_obj.forword(r, g)

        for e, r, m, tt, g in zip(
            batch["events"], batch["relations"], batch["mbtis"], batch["thens"], gens
        ):
            gen_data.append(
                {
                    "event": e,
                    "mbti": m,
                    "relation": r,
                    "ground": tt,
                    "generation": g,
                }
            )
            refs.append(tt)
            hyps.append(g)
    metric_result = metric_obj.close()

    results = {
        **metric_result,
        **calc_diversity(
            gen_data,
            dataset_name,
            train_file_path,
        ),
        "hyps": hyps,
        "refs": refs,
    }
    if not is_pure:
        results["loss"] = total_loss / total_samples
    model.train()
    return results


class CPCURTrainer(Trainer):
    def __init__(self, dataset_name: str, eval_metric_config: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.metric_obj = MyMetric(**eval_metric_config)

    def _save_checkpoint(self, eval_result: dict):
        steps = self._train_states["steps"]
        output_dir = os.path.join(self._config.output_dir, f"checkpoint-{steps}")
        os.makedirs(output_dir, exist_ok=True)

        # save
        self.model.plm.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)

        torch.save(
            self._model.state_dict(), os.path.join(output_dir, self.MODEL + ".pt")
        )
        torch.save(
            self._optimizer.state_dict(),
            os.path.join(output_dir, self.OPTIMIZER + ".pt"),
        )
        torch.save(
            self._train_states,
            os.path.join(output_dir, self.TRAIN_STATE + ".pt"),
        )
        if self._scheduler:
            torch.save(
                self._scheduler.state_dict(),
                os.path.join(output_dir, self.SCHEDULER + ".pt"),
            )
        LOGGER.info("Model saved at: ", output_dir)

        # update info
        self._train_states["best_golden_metric_value"] = eval_result[
            self._config.golden_metric
        ]
        self._train_states["best_ckpt_paths"].append(output_dir)

        if len(self._train_states["best_ckpt_paths"]) > self._config.save_total_limit:
            rm_dir(self._train_states["best_ckpt_paths"].pop(0))

    def build_train_loader(self, train_file_path: str):
        return DataLoader(
            dataset=CPCURDataset(train_file_path, self.dataset_name),
            batch_size=self.config.train_batch_size,
            shuffle=True,
        )

    def build_val_loader(self, val_file_path: str):
        return DataLoader(
            dataset=CPCURDataset(val_file_path, self.dataset_name),
            batch_size=self.config.eval_batch_size,
        )

    def evaluate_model(self):
        return evaluate_model(
            model=self.model,
            dataloader=self._val_loader,
            metric_obj=self.metric_obj,
            dataset_name=self.dataset_name,
            train_file_path=self._train_loader.dataset.file_path,
            is_pure=False,
        )

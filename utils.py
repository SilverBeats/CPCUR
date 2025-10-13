#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Dict
import warnings
import torch
import traceback
import jieba
from transformers import BartForConditionalGeneration, BartTokenizer
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from constant import MBTIS


def norm_string(s: str):
    return "".join(s.split())


def calc_diversity(
    gen_data: List[dict],
    dataset_name: str,
    train_file_path: str,
):
    if dataset_name == "atomic":
        npo = -1
    else:
        npo = 0
    # load all object from train file
    train_object_set = set(pd.read_csv(train_file_path)["then"].values.tolist())
    nto = 0
    gen_object = []
    for item in gen_data:
        gen_object.append(item["generation"])
        if item["generation"] not in train_object_set:
            nto += 1
    nto /= len(gen_object)
    nuo = len(set(gen_object)) / len(gen_object)

    if dataset_name != "atomic":
        _dict = {}
        for item in gen_data:
            key = f'{item["event"]} {item["relation"]}'
            if key not in _dict:
                _dict[key] = {k: [] for k in MBTIS}
            _dict[key][item["mbti"]].append(item["generation"])

        for er, v_dict in _dict.items():
            arr = []
            for p_r in v_dict.values():
                if len(p_r) >= 1:
                    arr.append(p_r[0])
            npo += len(set(arr)) / len(arr)

        npo /= len(_dict)

    return {"nto": nto, "npo": npo, "nuo": nuo}


class MyMetric:
    def __init__(
        self,
        use_nlgeval: bool = False,
        use_bert_score: bool = False,
        use_bart_score: bool = False,
        **kwargs,
    ):
        self.refs: List[List[str]] = []
        self.hyps: List[List[str]] = []

        self.use_nlgeval = use_nlgeval
        self.use_bert_score = use_bert_score
        self.use_bart_score = use_bart_score
        if use_nlgeval:
            from nlgeval import NLGEval

            print("create nlgeval obj")
            self.nlgeval = NLGEval(
                no_skipthoughts=True, no_glove=True, metrics_to_omit=["SPICE"]
            )

        if use_bart_score:
            print("load bart score obj")
            self.bart_score_config = kwargs.get("bert_score_config", {})
            self.bart_scorer = BARTScorer(**self.bart_score_config)

        if use_bert_score:
            self.bert_score_config = kwargs.get("bert_score_config", {})

    def forword(self, ref: str, hyp: str):
        reff, hypp = list(jieba.cut(ref.lower())), list(jieba.cut(hyp.lower()))
        if len(reff) != 0 and len(hypp) != 0:
            self.refs.append(reff)
            self.hyps.append(hypp)

    def calc_distinct_k(self, k) -> float:
        assert k >= 1
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, len(sen) - k):
                key = tuple(sen[i : i + k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn("the distinct is invalid")
            dist = 0.0
        return dist

    def close(self) -> Dict[str, float]:
        metric_res = {
            "length": float(np.mean(list(map(len, self.hyps)))),
            **{f"dist-{k}": 100 * self.calc_distinct_k(k) for k in range(1, 5)},
        }
        if self.use_bert_score:
            from bert_score import score

            print("use bert score")
            P, R, F = score(
                cands=["".join(item) for item in self.hyps],
                refs=["".join(item) for item in self.refs],
                **self.bert_score_config,
            )
            metric_res.update(
                {
                    "bert_score_P": round(P.mean().item(), 6),
                    "bert_score_R": round(R.mean().item(), 6),
                    "bert_score_F": round(F.mean().item(), 6),
                }
            )
        if self.use_nlgeval:
            print("use nlgeval")
            r = self.nlgeval.compute_metrics(
                ref_list=[[" ".join(item) for item in self.refs]],
                hyp_list=[" ".join(item) for item in self.hyps],
            )
            metric_res.update(r)

        if self.use_bart_score:
            print("use bart score")
            r = self.bart_scorer.score(self.refs, self.hyps)
            metric_res.update(r)

        return metric_res


class BARTScorer:
    """Code from https://github.com/neulab/BARTScore"""

    def __init__(
        self,
        device="cuda:0",
        max_length=1024,
        checkpoint="facebook/bart-large-cnn",
        path: str = None,
        batch_size: int = 32,
        **kwargs,
    ):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        self.batch_size = batch_size
        # Set up loss
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=1)

        if path is not None:
            self.load(path)

    def load(self, path):
        """Load model from paraphrase fine-tuning"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts) -> float:
        """Score a batch of examples"""
        score_list = []
        for i in tqdm(range(0, len(srcs), self.batch_size), desc="bart_score ..."):
            src_list = srcs[i : i + self.batch_size]
            tgt_list = tgts[i : i + self.batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        return np.mean(score_list, axis=0)

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean"):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self):
        """Test"""
        src_list = [
            "This is a very good idea. Although simple, but very insightful.",
            "Can I take a look?",
            "Do not trust him, he is a liar.",
        ]

        tgt_list = ["That's stupid.", "What's the problem?", "He is trustworthy."]

        print(self.score(src_list, tgt_list))

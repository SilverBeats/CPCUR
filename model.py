import torch.nn as nn
import torch
from transformers import GenerationConfig
import os
from typing import List, Optional, Union, Dict
from transformers import (
    BartForConditionalGeneration,
    BertLMHeadModel,
    BertTokenizer,
    GPT2LMHeadModel,
)
from constant import MBTIS, SPECIAL_TOKENS, PROMPT_TEMPLATE

CLS_MAP = {
    "bart": {
        "model": BartForConditionalGeneration,
        "tokenizer": BertTokenizer,
        "dim_key": "d_model",
    },
    "bert": {
        "model": BertLMHeadModel,
        "tokenizer": BertTokenizer,
        "dim_key": "hidden_size",
    },
    "gpt2": {"model": GPT2LMHeadModel, "tokenizer": BertTokenizer, "dim_key": "n_embd"},
}


class CSModel(nn.Module):
    def __init__(
        self,
        name_or_path: str,
        model_type: str,
        has_mbti: bool = True,
        generation_config: Optional[Union[GenerationConfig, dict]] = None,
    ):
        super().__init__()
        self.generation_config = {} if generation_config is None else generation_config

        self.has_mbti = has_mbti
        self.plm = CLS_MAP[model_type]["model"].from_pretrained(
            name_or_path,
            is_decoder=True,
        )
        self.tokenizer = CLS_MAP[model_type]["tokenizer"].from_pretrained(name_or_path)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.plm.resize_token_embeddings(len(self.tokenizer))

        d_model = getattr(self.plm.config, CLS_MAP[model_type]["dim_key"])

        self.fuse_linear = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
        )
        self.modify_tokenizer()

    def modify_tokenizer(self):
        self.tokenizer.padding_side = "left"
        self.tokenizer.eos_token_id = self.tokenizer.sep_token_id
        self.tokenizer.eos_token = self.tokenizer.sep_token

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.sep_token
            self.tokenizer.pad_token_id = self.tokenizer.sep_token_id

    @property
    def wte(self):
        return self.plm.get_input_embeddings()

    @property
    def device(self):
        return self.wte.weight.device

    def get_tokens_embeds(self, token_ids: List[int]) -> torch.FloatTensor:
        return self.wte(torch.LongTensor(token_ids).to(self.device))

    def fuse_relation_mbti(
        self, relations: List[str], mbtis: Optional[List[str]] = None
    ):
        batch_size = len(relations)
        # build mbti embedding
        # (batch_size, hidden size)
        if not self.has_mbti:
            mbti_embeds = (
                self.wte(
                    self.tokenizer(
                        text=MBTIS,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].to(self.device),
                )
                .mean(dim=1)
                .repeat_interleave(repeats=batch_size, dim=0)
            )
        else:
            mbti_embeds = self.wte(
                self.tokenizer(
                    text=mbtis,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(self.device),
            ).squeeze(1)

        # (batch_size, hidden size)
        relation_embeds = self.wte(
            self.tokenizer(
                text=relations,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(self.device),
        ).squeeze(1)

        # (batch_size, 1, hidden size)
        fuse_embeds = self.fuse_linear(
            torch.cat([mbti_embeds, relation_embeds], dim=-1),
        ).reshaoe(batch_size, 1, -1)
        return fuse_embeds

    def get_inputs_embeds(
        self,
        events: List[str],
        relations: List[str],
        mbtis: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        fuse_embeds = self.fuse_relation_mbti(relations, mbtis)
        event_ids = self.tokenizer(events, add_special_tokens=False)["input_ids"]
        each_len = [len(x) for x in event_ids]
        max_len = max(each_len)
        stack_arr = []
        batch_size = len(events)

        for i in range(batch_size):
            arr = [
                self.get_tokens_embeds([self.tokenizer.cls_token_id]),
                self.wte(torch.LongTensor(event_ids[i]).to(self.device)),
                fuse_embeds[i],
                self.get_tokens_embeds([self.tokenizer.convert_tokens_to_ids("[GEN]")]),
                self.get_tokens_embeds([self.tokenizer.sep_token_id]),
            ]
            if max_len - each_len[i] != 0:
                arr.append(
                    self.get_tokens_embeds(
                        [self.tokenizer.pad_token_id],
                    ).repeat_interleave(repeats=max_len - each_len[i], dim=0),
                )
            stack_arr.append(torch.cat(arr, dim=0))

        inputs_embeds = torch.stack(stack_arr, dim=0)
        src_mask = torch.stack(
            [
                torch.FloatTensor(
                    [1.0] * (inputs_embeds[i].shape[0] - max_len + each_len[i])
                    + [0.0] * (max_len - each_len[i]),
                )
                for i in range(batch_size)
            ],
            dim=0,
        ).to(self.device)

        position_ids = torch.cumsum(src_mask, dim=-1).long() - 1
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": src_mask,
            "position_ids": position_ids,
        }

    def prepare_p_k_v(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        model_kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "return_dict": True,
            "use_cache": True,
        }
        past_key_values = self.plm(**model_kwargs).past_key_values
        return past_key_values

    def prepare_y_for_train(self, ys: List[str]):
        y_input_ids = self.tokenizer(
            text=ys,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )["input_ids"].to(self.device)

        input_ids = y_input_ids[:, :-1].contiguous()
        lm_labels = y_input_ids[:, 1:].clone().detach()
        lm_labels[y_input_ids[:, 1:] == self.tokenizer.pad_token_id] = -100

        return input_ids, lm_labels

    def modify_generation_config(
        self, generation_config: Optional[Union[dict, GenerationConfig]] = None
    ):
        if generation_config is None:
            return GenerationConfig()
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        return generation_config

    @classmethod
    def from_pretrained(cls, name_or_path: str, has_mbti: bool):
        print(cls.MODEL_TYPE)
        model = cls(name_or_path, cls.MODEL_TYPE, has_mbti)
        if hasattr(model, "fuse_linear"):
            model.fuse_linear.load_state_dict(
                torch.load(os.path.join(name_or_path, "fuse.pt")),
            )
        return model

    @torch.no_grad()
    def generate(
        self,
        events: List[str],
        rels: List[str],
        mbtis: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        batch_size = len(events)

        generation_config = self.modify_generation_config(self.generation_config)

        inputs = self.get_inputs_embeds(events, rels, mbtis)
        p_k_v = self.prepare_p_k_v(**inputs)

        input_ids = (
            torch.empty((batch_size, 1))
            .fill_(self.tokenizer.cls_token_id)
            .long()
            .to(self.device)
        )
        position_ids = inputs["position_ids"][:, -1:] + 1
        attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones((batch_size, 1)).to(self.device)],
            dim=1,
        )
        gen_token_ids = self.plm.generate(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=p_k_v,
            generation_config=generation_config,
        )
        gen_tokens: List[str] = self.tokenizer.batch_decode(
            gen_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return gen_tokens

    def forward(
        self,
        events: List[str],
        relations: List[str],
        thens: List[str],
        mbtis: Optional[List[str]] = None,
        **kwargs
    ):
        inputs = self.get_inputs_embeds(events, relations, mbtis)
        past_key_values = self.prepare_p_k_v(**inputs)

        y_input_ids, label_ids = self.prepare_y_for_train(thens)

        tgt_mask = y_input_ids.ne(self.tokenizer.pad_token_id).float()
        tgt_mask = torch.cat([inputs["attention_mask"], tgt_mask], dim=1)
        tgt_position_ids = (
            torch.cumsum(tgt_mask, dim=-1).long()[:, -y_input_ids.shape[1] :] - 1
        )

        loss = self.plm(
            input_ids=y_input_ids,
            attention_mask=tgt_mask,
            position_ids=tgt_position_ids,
            past_key_values=past_key_values,
            labels=label_ids,
        ).loss
        return {"loss": loss}


class GPT2CSModel(CSModel):
    MODEL_TYPE = "gpt2"

    def __init__(
        self,
        name_or_path: str,
        has_mbti: bool,
        generation_config: Optional[Union[dict, GenerationConfig]] = None,
    ):
        super().__init__(name_or_path, self.MODEL_TYPE, has_mbti, generation_config)


class BertCSModel(CSModel):
    MODEL_TYPE = "bert"

    def __init__(
        self,
        name_or_path: str,
        has_mbti: bool,
        generation_config: Optional[Union[dict, GenerationConfig]] = None,
    ):
        super().__init__(name_or_path, self.MODEL_TYPE, has_mbti, generation_config)


class BartCSModel(CSModel):
    MODEL_TYPE = "bart"

    def __init__(
        self,
        name_or_path: str,
        has_mbti: bool,
        generation_config: Optional[Union[dict, GenerationConfig]] = None,
    ):
        super().__init__(name_or_path, self.MODEL_TYPE, has_mbti, generation_config)

    def forward(
        self,
        events: List[str],
        rels: List[str],
        thens: List[str],
        mbtis: Optional[List[str]] = None,
        **kwargs
    ):
        inputs = self.get_inputs_embeds(events, rels, mbtis)
        inputs.pop("position_ids")
        y_input_ids, label_ids = self.prepare_y_for_train(thens)
        loss = self.plm(**inputs, decoder_input_ids=y_input_ids, labels=label_ids).loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self, events: List[str], rels: List[str], mbtis: List[str] = None, **kwargs
    ):
        generation_config = self.modify_generation_config(self.generation_config)

        inputs = self.get_inputs_embeds(events, rels, mbtis)
        inputs.pop("position_ids")
        gen_token_ids = self.plm.generate(**inputs, generation_config=generation_config)

        gen_tokens: List[str] = self.tokenizer.batch_decode(
            gen_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return gen_tokens


class PureCSModel(CSModel):
    def __init__(
        self,
        model_type: str,
        pretrained_model_path: str,
        generation_config: Union[dict, GenerationConfig],
        **kwargs
    ):
        super().__init__(pretrained_model_path, model_type, False)
        del self.fuse_linear
        self.model_type = model_type
        self.generation_config = (
            generation_config if generation_config is not None else {}
        )

    @staticmethod
    def from_pretrained(model_type: str, pretrained_model_path: str, **kwargs):
        return PureCSModel(model_type, pretrained_model_path)

    @torch.no_grad()
    def generate(self, events: List[str], rels: List[str], **kwargs):
        generation_config = self.modify_generation_config(self.generation_config)

        new_events = []
        for e, r in zip(events, rels):
            new_events.append(PROMPT_TEMPLATE[r[1:-1]].format(e))

        inputs = self.tokenizer(
            text=new_events,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)
        if self.model_type != "bart":
            inputs["position_ids"] = (
                torch.cumsum(inputs["attention_mask"], dim=-1).long() - 1
            )
        ori_len = inputs["input_ids"].shape[1]
        gen_token_ids = self.plm.generate(**inputs, generation_config=generation_config)
        gens: List[str] = self.tokenizer.batch_decode(
            gen_token_ids[:, ori_len:] if self.model_type != "bart" else gen_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return gens

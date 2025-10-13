#!/usr/bin/env python3
# -*- coding: utf-8 -*-


RELS = [
    "[xAttr]",
    "[xWant]",
    "[xEffect]",
    "[xNeed]",
    "[xIntent]",
    "[xReact]",
    "[oReact]",
    "[oEffect]",
    "[oWant]",
]

MBTIS = [
    "[ISTJ]",
    "[ISTP]",
    "[ISFJ]",
    "[ISFP]",
    "[INTJ]",
    "[INTP]",
    "[INFJ]",
    "[INFP]",
    "[ESTJ]",
    "[ESTP]",
    "[ESFJ]",
    "[ESFP]",
    "[ENTJ]",
    "[ENTP]",
    "[ENFJ]",
    "[ENFP]",
]

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[PersonX]",
        "[PersonY]",
        "[MBTI]",
        "[GEN]",
        *RELS,
        *MBTIS,
    ]
}


PROMPT_TEMPLATE = {
    "xAttr": "{}。这件事发生之后，你觉得他是一个什么样的人？",
    "xIntent": "你觉得他为什么要做“{}”？",
    "xNeed": "你觉得他需要具备什么条件才能让”{}“发生？",
    "xEffect": "{}。你觉得这件事发生之后，对他产生什么影响？",
    "xWant": "{}。你觉得这件事发生之后，他想要做什么？",
    "xReact": "{}。你觉得这件事发生之后，他是什么心情？",
    "oReact": "{}。你觉得这件事发生之后，其他人是什么心情？",
    "oWant": "{}。你觉得这件事发生之后，其他人想要做什么？",
    "oEffect": "{}。你觉得这件事发生之后，对其他人产生什么影响？",
}

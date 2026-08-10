import os
import json
import jsonlines
from pathlib import Path
from enum import Enum


class NodeType(Enum):
    query = -1
    irrelevant_doc = 0
    relevant_doc = 1
    leaf = 2


PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_recomp_abstractive": 'Question: {question}\n Document: {context}\n Summary: ',
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
    "prompt_refiner_lora": "[INST]<<SYS>>[MONITOR]{context}<</SYS>>{question}[/INST] ",
    "prompt_refiner_prefix": "[INST]<<SYS>>{context}<</SYS>>{question}[/INST] ",
    "prompt_downstream_lora": "[INST]<<SYS>>[EXECUTOR]{refiner}<</SYS>>{question}[/INST] ",
    "prompt_downstream_prefix": "[INST]<<SYS>>{refiner}<</SYS>>{question}[/INST] ",
}

TRAIN_DATASET = {
    # self-rag
    "train": "wiki_retrieve.jsonl",
    "triviaqa_train": "triviaqa_train_retrieve.jsonl",
    "hotpotqa_train": "hotpotqa_train_processed.json",
    "2wiki_train": "2wiki_train_processed.json",
    "arc_c_train": "arc_c_train_retrieve.jsonl",
    "pubhealth_train": "pubhealth_train_processed.jsonl",
    "musique_train": "musique_train_processed.jsonl",
    "train_truncated": "arc_c_hotpotqa_triviaqa_truncated.jsonl",
}

TRAIN_DATASET_REFINER_OUTPUT = {
    # self-rag
    "train": "wiki_refiner_teacher_expunge.jsonl",
    "triviaqa_train": "triviaqa_refiner_teacher_expunge.jsonl",
    "hotpotqa_train": "hotpotqa_refiner_teacher_expunge.jsonl",
    "2wiki_train": "2wiki_refiner_teacher_expunge.jsonl",
    "arc_c_train": "arc_c_refiner_teacher_expunge.jsonl",
    "pubhealth_train": "pubhealth_refiner_teacher_expunge.jsonl",
    "musique_train": "musique_refiner_teacher_expunge.jsonl",
    "train_truncated": "llama3_truncated/arc_c_hotpotqa_triviaqa_truncated.jsonl",
}

EVAL_DATASET = {
    # short-form
    "arc_c": "arc_challenge_processed.jsonl",
    "triviaqa": "triviaqa_test.jsonl",
    "popqa": "popqa_dev_processed.jsonl",
    # multi-hop
    "hotpotqa_fullwiki_dev": "hotpotqa_fullwiki_dev_processed.json",
    "hotpotqa_distractor_dev": "hotpotqa_distractor_dev_processed.json",
    "hotpotqa_test": "hotpotqa_test_processed.json",
    "2wiki_dev": "2wiki_dev_processed.jsonl",
    "2wiki_test": "2wiki_test_processed.jsonl",
    "musique_dev": "musique_dev_processed.jsonl",
    "multihop_rag_dev": "multihop_rag_dev_processed.jsonl"
}

EVAL_DATASET_REFINER_OUTPUT = {
    # short-form
    "arc_c": "arc_c_refiner_teacher_expunge.jsonl",
    "triviaqa": "triviaqa_refiner_teacher_expunge.jsonl",
    "popqa": "popqa_refiner_teacher_expunge.jsonl",
    # multi-hop
    "hotpotqa_dev_fullwiki": "hotpotqa_dev_fullwiki_refiner_teacher_expunge.jsonl",
    "hotpotqa_dev_distractor": "hotpotqa_dev_distractor_refiner_teacher_expunge.jsonl",
    "hotpotqa_test": "hotpotqa_test_refiner_teacher_expunge.jsonl",
    "2wiki_dev": "2wiki_dev_refiner_teacher_expunge.jsonl",
    "2wiki_test": "2wiki_test_refiner_teacher_expunge.jsonl",
    "musique_dev": "musique_dev_refiner_teacher_expunge.jsonl"
}


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file_jsonl(input_file_path: str | Path):
    input_file_path = str(input_file_path)
    if input_file_path.endswith(".json"):
        with open(input_file_path) as f:
            input_data = json.load(f)
    else:
        input_data = load_jsonlines(input_file_path)
    return input_data


def save_file_jsonl(data, output_file_path: str | Path, mode='w'):
    output_file_path = str(output_file_path)
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    if output_file_path.endswith("json"):
        with open(output_file_path) as f:
            json.dump(data, f)
    else:
        with jsonlines.open(output_file_path, mode=mode) as f:
            f.write_all(data)


def load_file(input_fp):
    input_fp = str(input_fp)
    if input_fp.endswith(".json"):
        with open(input_fp) as f:
            input_data = json.load(f)
    else:
        input_data = load_jsonlines(input_fp)
    return input_data
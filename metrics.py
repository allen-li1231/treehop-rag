import re
import numpy as np
import torch
import torch.nn.functional as F

from utils import NodeType
from tree_hop.model import graph_cosine_similiarity
from src.evaluation import normalize_answer

regex_answer_key = re.compile(r"is[^\w]+([A-E])[^\w]+", re.DOTALL)


def gather_graph_similarity_score(g, normalize=True):
    sim = graph_cosine_similiarity(g, normalize=normalize)
    positives = torch.where(g.ndata["y"] >= NodeType.relevant_doc.value)[0].int()
    negatives = torch.where(g.ndata["y"] == NodeType.irrelevant_doc.value)[0].int()
    gain = sim[g.in_edges(positives, form="eid")].mean()
    loss = sim[g.in_edges(negatives, form="eid")].mean()
    return (gain - loss).item()


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))


def find_entity_tags(sentence):
    entity_regex = r'(.+?)(?=\s<|$)'
    tag_regex = r'<(.+?)>'
    entity_names = re.findall(entity_regex, sentence)
    tags = re.findall(tag_regex, sentence)

    results = {}
    for entity, tag in zip(entity_names, tags):
        if "<" in entity:
            results[entity.split("> ")[1]] = tag
        else:
            results[entity] = tag
    return results

def match(prediction, ground_truth):
    prediction = prediction.lower()
    for gt in ground_truth:
        if gt.lower() in prediction:
            return 1
    return 0


def calc_acc(data, context: str):
    if "answers" in data:
        context = normalize_answer(context)
        return any(a.lower() in context for a in data["answers"])

    elif "answerKey" in data:
        if len(data["answerKey"]) == 0:
            return re.search(regex_answer_key, context) is None

        lst_res = []
        for key in data["answerKey"]:
            search = re.search(regex_answer_key, context)
            if search is None:
                lst_res.append(False)
                continue

            if key.lower() == search.groups()[0].lower():
                lst_res.append(True)
            else:
                lst_res.append(False)

        return all(lst_res)

    elif "answer" in data:
        context = normalize_answer(context)
        answer = normalize_answer(data["answer"])
        return answer in context
    elif isinstance(data, str):
        context = normalize_answer(context)
        answer = normalize_answer(data)
        return answer in context
import re
import string
import collections
import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


# Exact match (the normalized answer exactly match the gold answer)
def exact(predictions, references):
    return int(normalize_answer(references[0]) == normalize_answer(predictions[0]))

def solution_present(predictions, references):
    return int(normalize_answer(references[0]) in normalize_answer(predictions[0]))


# The F-score of predicted tokens versus the gold answer
def f1(prediction, reference):
    gold_toks = get_tokens(reference)
    pred_toks = get_tokens(prediction)

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    
    #common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    common_tokens = set(pred_toks) & set(gold_toks)
    num_same = len(common_tokens)
    
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1(items):
    golds, preds = zip(*items)
    sum = 0
    for gold, pred in items:
        sum += f1(pred, gold)
    return sum/len(golds)

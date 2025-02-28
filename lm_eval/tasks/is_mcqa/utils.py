import datasets
import re

def subset_accuracy(references, predictions):
    correct_count = 0
    for correct, pred in zip(references, predictions):
        if set(correct) == set(pred):
            correct_count += 1
    return correct_count / len(references)


def jaccard_index(references, predictions):
    jaccard_scores = []
    for correct, pred in zip(references, predictions):
        intersection = len(set(correct) & set(pred))
        union = len(set(correct) | set(pred))
        jaccard_scores.append(intersection / union)
    return sum(jaccard_scores) / len(jaccard_scores)


def get_answers(row):
    answers_list = []
    answer = row["answer"]
    matches = re.findall(r"(?:The answer is: |The answers are: )([A-Z](?:,\s*[A-Z])*)", answer)
    if matches:
        match = matches[0]
        # Split on spaces and strip
        answers_list = [ans.strip() for ans in match.split(",")]
    return {"answers": answers_list}


def process_dataset(dataset: datasets.Dataset):
    dataset.map(get_answers)
    return dataset

def process_results(doc: datasets.Dataset, results):
    preds = results[0]
    references = doc["answers"]

    print('preds:', preds)
    print('references:', references)
    # Process preds
    preds = [get_answers(pred) for pred in preds]
    # Compute metrics
    subset_acc = subset_accuracy(references, preds)
    jaccard = jaccard_index(references, preds)

    return {"acc": subset_acc, "IoU": jaccard}



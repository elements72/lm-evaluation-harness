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

    if len(jaccard_scores) == 0:
        return 0
    return sum(jaccard_scores) / len(jaccard_scores)


def process_answer(answer):
    # Split on commas
    answer = answer.split("Q: ")[-1].strip()
    return answer


def process_results(doc: datasets.Dataset, results):
    print(results)
    preds = results[0]
    references = doc["answer"]

    # Process preds
    pred = process_answer(preds)
    # Compute metrics


    print('Question:', doc['question'])
    print('Gold:', references)
    print('Model out:', results[0])
    print('Processed out:', preds)
    print('Subset Acc:', subset_acc)
    print('Jaccard:', jaccard)

    return {"acc": subset_acc, "IoU": jaccard}


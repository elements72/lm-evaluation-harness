import datasets
import re
import evaluate

from scripts.model_comparator import print_results

bertscore = evaluate.load("bertscore")

def process_answer(answer):
    # Split on commas
    answer = answer.split("Q: ")[-1].strip()
    return answer


def process_results(doc, results):
    print("doc:", doc)
    print("results:", results)
    results = results[0]
    result = bertscore.compute(predictions=results, references=doc)

    return {"bertscore_f1": result['f1'][0], "bertscore_precision": result['precision'][0], "bertscore_recall": result['recall'][0]}


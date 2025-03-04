import datasets
import re
import evaluate

bertscore = evaluate.load("bertscore", lang="en")

def process_answer(answer):
    # Split on commas
    answer = answer.split("Q: ")[-1].strip()
    return answer


def process_results(doc, results):
    print("doc:", doc)
    print("results:", results)
    result = bertscore.compute(predictions=results, references=[doc['answer']], lang="en")

    return {"bertscore_f1": result['f1'][0], "bertscore_precision": result['precision'][0], "bertscore_recall": result['recall'][0]}


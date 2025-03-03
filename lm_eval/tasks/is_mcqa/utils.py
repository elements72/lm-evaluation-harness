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


def map_to_answers(row):
    return {'answers': get_answers(row['output']), 'output': ', '.join(get_answers(row['output']))}


def doc_to_text(doc):
    string = f'{doc["input"]}'

    if 'first' in doc.keys():
        string = f'The following are multiple choice questions (with answers) about EO.\n' + string

    return string

def get_answers(answer):
    answers_list = []
    matches = re.findall(r"(?:The answer is: |The answers are: )([A-Z](?:,\s*[A-Z])*)", answer)
    if matches:
        match = matches[0]
        # Split on spaces and strip
        answers_list = [ans.strip() for ans in match.split(",")]
    return answers_list


def process_dataset(dataset: datasets.Dataset):
    return dataset.map(map_to_answers)

def process_results(doc: datasets.Dataset, results):
    print(results)
    preds = results[0]
    references = doc["answers"]

    print('doc:', doc)
    print('preds:', preds)
    print('references:', references)
    # Process preds
    preds = [get_answers(pred) for pred in preds]
    # Compute metrics
    subset_acc = subset_accuracy(references, preds)
    jaccard = jaccard_index(references, preds)

    return {"acc": subset_acc, "IoU": jaccard}


def list_fewshot_samples() -> list[dict]:
    return [
        {"input": """
        What are the impacts of a warming climate on the cryosphere?

        A) Global warming causes melting of ice on land such as glaciers, and ice sheets in Antarctica and Greenland. This adds water to the ocean, and raises global sea levels
        B) Ice and snow reflect sunlight and have a cooling effect on the climate. Without ice, darker land and ocean absorbs heat and amplifies climate change.
        C) Thawing permafrost releases significant amounts of greenhouse gases, causing further warming
        D) Human populations living off or close to the sea will be affected as well as population within mountain regions
                
        """, "first": True, 'output': 'The answers are: A, B, C, D'},

        {"input": """What is ice and snow albedo?

        A) Albedo is the measure of how much light that hits a surface is reflected without being absorbed
        B) Albedo is the measure the thickness of ice or snow
        C) Albedo is the temperature of ice or snow
        D) Albedo is the ratio between frozen water and snow
                
        """, 'output': 'The answer is: A'},
    ]

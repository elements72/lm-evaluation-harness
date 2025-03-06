import datasets
import re
import evaluate
from openai import OpenAI
import time
import os

bertscore = evaluate.load("bertscore", lang="en")
client = OpenAI()



def create_chat_prompt(question: str, llm_answer: str, answer: str) -> list[dict[str, str]]:
    sys_msg = """Evaluate the answer of a AI model to a question. You will be provided with the question, the AI model’s answer, and the correct answer. Your task is to evaluate the AI model’s response and determine whether it is Correct or Incorrect.
            Grade the AI model answers based ONLY on their factual accuracy. It is OK if the AI model answer contains more information than the true answer, as long as it does not contain any conflicting statements. Otherwise, it should be marked as Incorrect. Ignore differences in punctuation and phrasing between the AI model’s answer and the true answer.
            Example Format:
            QUESTION: question here
            AI ANSWER: AI answer here
            TRUE ANSWER: true answer here
            GRADE: Correct or Incorrect here
            Your response should include only the verdict without any justification or reasoning"""
    user_prompt = f"""QUESTION: {question}\n AI ANSWER: {llm_answer}\n TRUE ANSWER: {answer}\n GRADE: """
    #print(user_prompt)
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_prompt}
    ]


def process_answer(answer):
    # Split on commas
    answer = answer.split("Q: ")[-1].strip()
    return answer


def get_chat_completion(prompt: list[dict[str, str]]) -> str:
    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(messages=prompt,
                                                        model='gpt-4-turbo')
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            attempt += 1
            time.sleep(3)
    print(f"Failed to get chat completion after {max_retries} attempts")
    return "Cannot be answered"


def llm_as_judge(question: str, llm_answer: str, answer: str) -> dict[str, int]:
    prompt = create_chat_prompt(question, llm_answer, answer)
    llm_judge = get_chat_completion(prompt)
    #print(f"LM: {llm_judge}")
    # Parse Yes/No into 1, 0 for accuracy
    llm_judge = 0 if 'incorrect' in llm_judge.lower() else 1

    return {"llm_judge_accuracy": llm_judge}


def bertscore_metric(predictions: list[str], references: list[str], threshold=0.50) -> dict[str, float]:
    result = bertscore.compute(predictions=predictions, references=references, lang="en")
    f1 = result['f1'][0]
    precision = result['precision'][0]
    recall = result['recall'][0]
    accuracy = 1 if f1 > threshold else 0
    result = {'bertscore_f1': f1, 'bertscore_precision': precision, 'bertscore_recall': recall, 'bertscore_accuracy': accuracy}
    return result




def process_results(doc, results):
    print("doc:", doc)
    print("results:", results)
    dict_results = {}
    bertscore_results = bertscore.compute(predictions=results, references=[doc['answer']], lang="en")
    dict_results.update(bertscore_results)
    llm_result = llm_as_judge(doc['question'], results[0], doc['answer'])
    dict_results.update(llm_result)
    print("dict_results:", dict_results)
    return dict_results



import sqlite3
from typing import Dict, List

from recruit_ai.application.question_result_obtain import \
    extract_number_from_string
from recruit_ai.application.tag_obtain import tag_obtain
from recruit_ai.data_utils.request_model import request_model


def eval_fill_in_questions(
    fill_in_questions: List[Dict[str, str]],
    candidate_answers: List[str],
    model: str='Qwen2-7B-Instruct'
) -> List[float]:
    result = []
    for question, answer in zip(fill_in_questions, candidate_answers):
        model_response = request_model(model,
                                       'eval_fill_in',
                                       question=question['question'],
                                       correct_answer=question['answer'],
                                       candidate_answer=answer)
        result.append(extract_number_from_string(model_response))
    return result


def eval_choice_or_judge_questions(
    choice_or_judge_questions: List[Dict[str, str | List[str]]],
    candidate_answers: List[str],
) -> List[float]:
    result = []
    for question, answer in zip(choice_or_judge_questions, candidate_answers):
        result.append(10 if question['answer'] == answer else 0)
    return result


def select_category_labels_from_db(
    c: sqlite3.Cursor,
    table: str,
    id: int,
) -> Dict[str, str | List[str]]:
    c.execute(f'SELECT category, labels FROM {table} WHERE id = ?', (id, ))
    results = c.fetchall()
    if results:
        return {'category': results[0][0], 'labels': results[0][1].split(',')}
    else:
        tag_obtain(table[len('tagged_'):], id, True)
        c.execute(f'SELECT category, labels FROM {table} WHERE id = ?', (id, ))
        results = c.fetchall()
        return {'category': results[0][0], 'labels': results[0][1].split(',')}

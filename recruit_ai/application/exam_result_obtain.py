import json
import re
from datetime import datetime
from traceback import format_exc
from typing import Any, Dict, List, Tuple

from recruit_ai.application.question_obtain import \
    select_category_labels_from_db
from recruit_ai.data_utils import sql_utils
from recruit_ai.data_utils.data_manager import PATHS, load_data_by_id
from recruit_ai.data_utils.request_model import request_model


def extract_number_from_string(s: str) -> int:
    numbers = re.findall(r'\d+', s)
    if numbers:
        num = int(numbers[0])
        return num if 0 <= num and num <= 10 else 0
    else:
        return 0


def contains_chinese_or_english(s: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', s))


def eval_fill_in_questions(
    fill_in_questions: List[Dict[str, str]],
    candidate_answers: List[str],
) -> Tuple[List[int], List[int]]:
    scores = []
    real_scores = []
    for question, answer in zip(fill_in_questions, candidate_answers):
        if not answer or answer.isspace() or (
                contains_chinese_or_english(question['answer'])
                and not contains_chinese_or_english(answer)):
            score = 0
            print('invalid answer')
            print(question['answer'])
        else:
            model_response = request_model('Qwen2-7B-Instruct',
                                           'eval_fill_in',
                                           question=question['question'],
                                           correct_answer=question['answer'],
                                           candidate_answer=answer)
            print(question['answer'])
            score = extract_number_from_string(model_response)
        scores.append(score)
        real_scores.append(score * question['score'] // 10)
    return scores, real_scores


def eval_choice_or_judge_questions(
    choice_or_judge_questions: List[Dict[str, str | List[str]]],
    candidate_answers: List[str],
) -> Tuple[List[float], List[int]]:
    scores = []
    real_scores = []
    for question, answer in zip(choice_or_judge_questions, candidate_answers):
        if question['answer'] == answer:
            scores.append(10)
            real_scores.append(question['score'])
        else:
            scores.append(0)
            real_scores.append(0)
    return scores, real_scores


def calculate_question_label_score(
    questions: List[Dict[str, str | List[str]]],
    scores: List[str],
) -> Dict[str, float]:
    label_score = {}
    label_count = {}
    for question, score in zip(questions, scores):
        for label in question['labels']:
            if label not in label_score:
                label_score[label] = 0
                label_count[label] = 0
            label_score[label] += score
            label_count[label] += 1
    for k in label_score:
        label_score[k] /= label_count[k]
    return dict(
        sorted(label_score.items(), key=lambda item: item[1], reverse=True))


def calculate_jd_match_degree(
    jds: List[Dict[str, int | str | List[str]]],
    label_score: Dict[str, float],
) -> List[int]:
    result = []
    for jd in jds:
        match_degree = 0
        cnt = 0
        for label in jd['labels']:
            if label in label_score:
                match_degree += label_score[label]
                cnt += 1
        if cnt == 0:
            result.append(0)
        else:
            match_degree = match_degree / cnt * 10  # Converted to 100 points
            match_degree = round(match_degree / 10) * 10  # Round to ten
            result.append(match_degree)
    return result


def analyse_behaviour_by_score(score: int) -> Tuple[str, str]:
    if score <= 70:
        if score <= 60:
            return '较差', '低（0-60）'
        else:
            return '较差', '中低（61-70）'
    elif score <= 85:
        return '一般', '中（71-85）'
    else:
        return '出色', '高（86-100）'


def filt_label_score(
    questions: List[Dict[str, str | List[str]]],
    label_score: Dict[str, float],
    jd_labels: List[str],
) -> Dict[str, float]:
    label_question_cnt = {}
    for question in questions:
        for label in question['labels']:
            if label not in label_question_cnt:
                label_question_cnt[label] = 0
            label_question_cnt[label] += 1
    print(label_question_cnt)
    print(jd_labels)
    filted_label_score = {}
    for label in jd_labels:
        if label in label_question_cnt and label_question_cnt[label] >= 3:
            filted_label_score[label] = label_score.get(label)
    return sorted(filted_label_score.items(),
                  key=lambda item: item[1],
                  reverse=True)


def load_question_and_answer_set(question_set_id: int):
    row1 = sql_utils.select_question_set_from_que_db(question_set_id)
    choice_questions = []
    choice_ids = json.loads(row1[0])
    for choice_id in choice_ids:
        row2 = sql_utils.select_choice_from_que_db(choice_id)
        choice_questions.append({
            'question': row2[1],
            'options': json.loads(row2[2]),
            'answer': row2[3],
            'labels': row2[0].split(',')
        })
    fill_bank_questions = []
    fill_bank_ids = json.loads(row1[1])
    for fill_bank_id in fill_bank_ids:
        row2 = sql_utils.select_fill_bank_from_que_db(fill_bank_id)
        fill_bank_questions.append({
            'question': row2[1],
            'answer': row2[2],
            'labels': row2[0].split(',')
        })
    judge_questions = []
    judge_ids = json.loads(row1[2])
    for judge_id in judge_ids:
        row2 = sql_utils.select_judge_from_que_db(judge_id[0])
        judge_questions.append({
            'question': json.loads(row2[1])[judge_id[1]],
            'answer': json.loads(row2[2])[judge_id[1]],
            'labels': row2[0].split(',')
        })
    return {
        'fill_in_questions': fill_bank_questions,
        'choice_questions': choice_questions,
        'true_or_false_questions': judge_questions
    }


def add_per_question_score(
    question_and_answer_set: Dict[str, List[List[Dict[str, Any]]]]
) -> Dict[str, List[List[Dict[str, Any]]]]:
    ret = {}
    question_num = 0
    for questions in question_and_answer_set.values():
        question_num += len(questions)
    score = 100 // question_num
    cnt = 0
    for type, questions in question_and_answer_set.items():
        for question in questions:
            question['score'] = score
            if cnt < 100 % question_num:
                question['score'] += 1
            cnt += 1
        ret[type] = questions
    return ret


def question_result_obtain_(
    cv_id: int,
    need_idset: str,
    question_set_id: int,
    answer_set: Dict[str, List[str]],
) -> Dict[str, Any]:
    try:
        need_jd_ids = need_idset.split(',')
        jds = []
        for i, jd_id in enumerate(need_jd_ids):
            jds += load_data_by_id(PATHS['jd'], '需求编号', int(jd_id))
            category_labels = select_category_labels_from_db(
                'tagged_jd', int(jd_id))
            jds[i]['category'] = category_labels['category']
            jds[i]['labels'] = category_labels['labels']
        print(jds)
        question_and_answer_set = load_question_and_answer_set(question_set_id)
        question_and_answer_set = add_per_question_score(
            question_and_answer_set)
        print(question_and_answer_set)
        questions = question_and_answer_set[
            'fill_in_questions'] + question_and_answer_set[
                'choice_questions'] + question_and_answer_set[
                    'true_or_false_questions']
        global score_per_question
        score_per_question = 100 / len(questions)
        print('Evaluating question answers...')
        fill_in_scores, real_fill_in_scores = eval_fill_in_questions(
            question_and_answer_set['fill_in_questions'],
            answer_set['fill_in_questions'],
        )
        choice_scores, real_choice_scores = eval_choice_or_judge_questions(
            question_and_answer_set['choice_questions'],
            answer_set['choice_questions'],
        )
        judge_scores, real_judge_scores = eval_choice_or_judge_questions(
            question_and_answer_set['true_or_false_questions'],
            answer_set['true_or_false_questions'],
        )
        label_score = calculate_question_label_score(
            questions, fill_in_scores + choice_scores + judge_scores)
        match_degrees = calculate_jd_match_degree(jds, label_score)
        max_match_degree = max(match_degrees)
        need_id = int(need_jd_ids[match_degrees.index(max_match_degree)])
        test_score = int(
            sum(real_fill_in_scores + real_choice_scores + real_judge_scores))
        behaviour = analyse_behaviour_by_score(test_score)
        analysis_explanation = f'根据笔试成绩显示，该候选人表现{behaviour[0]}，{test_score}分的成绩位于{behaviour[1]}分段。'
        jd_labels = []
        for jd in jds:
            jd_labels += jd['labels']
        filted_label_score = filt_label_score(questions, label_score,
                                              jd_labels)
        print(filted_label_score)
        if len(filted_label_score) > 0:
            score_to_comment = int(
                min(filted_label_score[0][1] / score_per_question * 100, 100))
            comment = request_model('Qwen2-7B-Instruct',
                                    'comment_highest_score',
                                    label_to_comment=filted_label_score[0][0],
                                    score_to_comment=score_to_comment)
            if score_to_comment < 60:
                analysis_explanation += f'候选人仅在{filted_label_score[0][0]}方面表现相对较好，{comment}\n'
            else:
                analysis_explanation += f'候选人在{filted_label_score[0][0]}方面表现较好，{comment}\n'
        if max_match_degree < 60:
            analysis_explanation += '根据笔试情况，该候选人与所有岗位匹配度均较低，没有匹配的岗位。\n'
        else:
            jd_idx = match_degrees.index(max_match_degree)
            analysis_explanation += f'根据笔试情况，在与其可能匹配的岗位中，推荐匹配度最高的是{jds[jd_idx]["需求岗位"]}，匹配度为{max_match_degree}。因而推荐候选人与该岗位适配。\n'
        return {
            'message': 'success',
            'result': {
                'need_id': need_id,
                'matching_degree': max_match_degree,
                'test_score': test_score,
                'analysis_explanation': analysis_explanation,
                'answer_score_set': {
                    'fill_in_questions': real_fill_in_scores,
                    'choice_questions': real_choice_scores,
                    'true_or_false_questions': real_judge_scores
                },
                'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            },
            'status': 0
        }
    except:
        result = dict()
        result["message"] = format_exc()
        result["result"] = dict()
        result["result"]["need_id"] = -1
        result["result"]["matching_degree"] = -1
        result["result"]["test_score"] = -1
        result["result"]["analysis_explanation"] = ""
        result["result"]["answer_score_set"] = {}
        result["result"]["time"] = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        result["status"] = -1
        return result

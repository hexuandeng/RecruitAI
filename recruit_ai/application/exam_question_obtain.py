import json
from datetime import datetime
from random import sample, seed, shuffle
from time import time
from traceback import format_exc
from typing import Any, Dict, List, Optional, Tuple

from recruit_ai.application.tag_obtain import tag_datas, tag_obtain
from recruit_ai.data_utils import sql_utils
from recruit_ai.data_utils.data_manager import PATHS, load_data_by_id
from recruit_ai.data_utils.make_proportion import make_proportion

with open(PATHS['label_map'], 'r', encoding='utf-8') as f:
    label_map = json.load(f)


def get_group_label(
    datas: List[Dict[str, int | str]],
    label_map: Dict[str, List[str]],
) -> List[Dict[str, List[str]]]:
    result = []
    for data in datas:
        result.append({
            'group':
            [data['category']] if data['category'] in label_map else [],
            'label':
            data['labels']
        })
    return result


def distribute_question(
    question_num_dict: Dict[str, int],
    question_nums: List[int],
) -> List[Dict[str, int]]:
    question_label = []
    for k, v in question_num_dict.items():
        for i in range(v):
            question_label.append(k)
    idx_list = [i for i in range(sum(question_nums))]
    result = []
    for question_num in question_nums:
        sub_idx_list = sample(idx_list, question_num)
        label_question_num = {}
        for i in sub_idx_list:
            try:
                if question_label[i] in label_question_num:
                    label_question_num[question_label[i]] += 1
                else:
                    label_question_num[question_label[i]] = 1
            except:
                if '软件工程' in label_question_num:
                    label_question_num['软件工程'] += 1
                else:
                    label_question_num['软件工程'] = 1
            idx_list.remove(i)
        result.append(label_question_num)
    return result


difficultys = ['初级', '中级', '高级']


def get_choice_quesion(
    question_ids: List[int],
    difficulty: str,
) -> List[Dict[str, str | List[str]]]:
    result = []
    for question_id in question_ids:
        row = sql_utils.select_choice_question_from_que_db(
            question_id, difficulty)
        if not row:
            for d in difficultys:
                row = sql_utils.select_choice_question_from_que_db(
                    question_id, d)
                if row:
                    break
        result.append({
            'question': row[0],
            'options': json.loads(row[1]),
            'answer': row[2]
        })
    return result


def get_fill_bank_question(
    question_ids: List[int],
    difficulty: str,
) -> List[Dict[str, str | List[str]]]:
    result = []
    for question_id in question_ids:
        row = sql_utils.select_fill_bank_question_from_que_db(
            question_id, difficulty)
        if not row:
            for d in difficultys:
                row = sql_utils.select_fill_bank_question_from_que_db(
                    question_id, d)
                if row:
                    break
        result.append({'question': row[0], 'answer': row[1]})
    return result


def get_judge_question(
    question_ids: List[int],
    difficulty: str,
) -> Tuple[List[Dict[str, str]], List[int]]:
    result = []
    judge_ids = []
    yes_questions = {}
    no_questions = {}
    only_yes = set()
    only_no = set()
    for question_id in question_ids:
        row = sql_utils.select_judge_question_from_que_db(
            question_id, difficulty)
        if not row:
            for d in difficultys:
                row = sql_utils.select_judge_question_from_que_db(
                    question_id, d)
                if row:
                    break
        questions = json.loads(row[0])
        answers = json.loads(row[1])
        for idx, answer in enumerate(answers):
            if answer == '是':
                if question_id not in yes_questions:
                    yes_questions[question_id] = []
                yes_questions[question_id].append((questions[idx], '是', idx))
            else:
                if question_id not in no_questions:
                    no_questions[question_id] = []
                no_questions[question_id].append((questions[idx], '否', idx))
        if question_id in yes_questions:
            shuffle(yes_questions[question_id])
        else:
            only_no.add(question_id)
        if question_id in no_questions:
            shuffle(no_questions[question_id])
        else:
            only_yes.add(question_id)
    x = 0
    y = 0
    for question_id in question_ids:
        if question_id in only_yes:
            x += 1
            result.append({
                'question': yes_questions[question_id][0][0],
                'answer': yes_questions[question_id][0][1]
            })
            judge_ids.append(yes_questions[question_id][0][2])
        elif question_id in only_no:
            y += 1
            result.append({
                'question': no_questions[question_id][0][0],
                'answer': no_questions[question_id][0][1]
            })
            judge_ids.append(no_questions[question_id][0][2])
        else:
            if x < y:
                x += 1
                result.append({
                    'question': yes_questions[question_id][0][0],
                    'answer': yes_questions[question_id][0][1]
                })
                judge_ids.append(yes_questions[question_id][0][2])
            else:
                y += 1
                result.append({
                    'question': no_questions[question_id][0][0],
                    'answer': no_questions[question_id][0][1]
                })
                judge_ids.append(no_questions[question_id][0][2])
    return result, judge_ids


def get_question(
    label_question_num: Dict[str, int],
    question_type: str,
    difficulty: str,
) -> Tuple[List[Dict[str, str | List[str]]], List[int]]:
    result = []
    all_question_ids = []
    for k, v in label_question_num.items():
        row = sql_utils.select_question_from_que_db(question_type, k)
        if not row or not row[0]:
            row = sql_utils.select_question_from_que_db_randomly(question_type)
        print(row[0])
        full_question_ids = [int(i) for i in row[0].split(',')]
        question_ids = sample(full_question_ids, min(v,
                                                     len(full_question_ids)))
        if question_type == 'choice':
            result += get_choice_quesion(question_ids, difficulty)
            all_question_ids += question_ids
        elif question_type == 'fill_bank':
            result += get_fill_bank_question(question_ids, difficulty)
            all_question_ids += question_ids
        elif question_type == 'judge':
            sub_result, sub_judge_ids = get_judge_question(
                question_ids, difficulty)
            result += sub_result
            all_question_ids += [[i, j]
                                 for i, j in zip(question_ids, sub_judge_ids)]
    return result, all_question_ids


def select_category_labels_from_db(
    table: str,
    id: int,
    update: bool = True,
) -> Dict[str, str | List[str]]:
    if update:
        tag_obtain(table[len('tagged_'):], id, update)
        row = sql_utils.select_category_labels_from_tag_db(table, id)
        return {'category': row[0], 'labels': row[1].split(',')}
    row = sql_utils.select_category_labels_from_tag_db(table, id)
    if row:
        return {'category': row[0], 'labels': row[1].split(',')}
    tag_obtain(table[len('tagged_'):], id, update)
    row = sql_utils.select_category_labels_from_tag_db(table, id)
    return {'category': row[0], 'labels': row[1].split(',')}


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


def get_jd_difficulty(jd: dict[str, int | str]) -> str:
    for d in ['初级', '中级', '高级']:
        if d in jd['post_level']:
            return d
    return '中级'


def question_obtain(
    cv_id: int,
    need_idset: str,
    project_experience: int,
    technical_level: int,
    fill_in_question_num: int,
    choice_question_num: int,
    true_or_false_question_num: int,
    other_requirements: Optional[str] = None,
    update: bool = False,
) -> Dict[str, str | int | Dict]:
    try:
        if project_experience + technical_level != 100:
            raise ValueError(
                'The sum of project_experience and technical_level must be 100.'
            )
        need_jd_ids = need_idset.split(',')
        jds = [
            select_category_labels_from_db('tagged_jd', int(jd_id), update)
            for jd_id in need_jd_ids
        ]
        tmp_jds = load_data_by_id(PATHS['jd'], '需求编号', int(
            need_jd_ids[0]))  # This step is to read the difficulty of the jd added later. There is no difficulty field in the original database design.
        difficulty = get_jd_difficulty(tmp_jds[0])
        print('Taging other requirements...')
        other_datas = [{'text': other_requirements}]
        tag_datas(other_datas, 'text', {'text': 'text'}, label_map)
        labels = [
            get_group_label(jds, label_map),
            get_group_label(other_datas, label_map)
        ]
        for i in labels:
            print(i)
        print('Generating questions...')
        question_num_dict = make_proportion(
            labels, fill_in_question_num + choice_question_num +
            true_or_false_question_num,
            [technical_level / 2 + 50, project_experience / 2, 20])
        label_question_nums = distribute_question(question_num_dict, [
            fill_in_question_num, choice_question_num,
            true_or_false_question_num
        ])
        question_types = ['fill_bank', 'choice', 'judge']
        questions = []
        type_question_ids = {}
        for label_question_num, question_type in zip(label_question_nums,
                                                     question_types):
            sub_questions, question_ids = get_question(label_question_num,
                                                       question_type,
                                                       difficulty)
            questions.append(sub_questions)
            type_question_ids[question_type] = question_ids
        end_time = time()
        question_and_answer_set = {
            'fill_in_questions': questions[0],
            'choice_questions': questions[1],
            'true_or_false_questions': questions[2]
        }
        result = {
            'answering_time':
            int((fill_in_question_num + choice_question_num +
                 true_or_false_question_num) * 1.5),
            'question_set_id':
            int(end_time),
            'question_and_answer_set':
            add_per_question_score(question_and_answer_set),
            'time':
            datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        sql_utils.insert_data_into_que_db(
            result['question_set_id'],
            json.dumps(type_question_ids['choice'], ensure_ascii=False),
            json.dumps(type_question_ids['fill_bank'], ensure_ascii=False),
            json.dumps(type_question_ids['judge'], ensure_ascii=False))
        return {'message': 'success', 'result': result, 'status': 0}
    except:
        result = dict()
        result["message"] = format_exc()
        result["result"] = dict()
        result["result"]["answering_time"] = -1
        result["result"]["question_set_id"] = -1
        result["result"]["question_and_answer_set"] = {}
        result["result"]["time"] = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        result["status"] = -1
        return result


def get_interview_question_id(group: str):
    row = sql_utils.select_interview_question_id_from_que_db(group)
    return str(row[0]).split(",") if row else []


def get_interview_question(id: int):
    row = sql_utils.select_interview_question_answer_from_que_db(id)
    # print(f"==========={row}============")
    return {"question": row[0], "answer": row[1], "difficult": row[2]}


def get_interview_question_with_difficulty(questions_list, difficult_jd,
                                         label_nums):
    # Shuffle the question list to randomly sample data
    shuffle(questions_list)
    matching_difficult = []  
    other_cases = []  

    all_value = sum(label_nums.values())  # Total required number of cases
    # Early exit condition: stop once we have enough matching cases
    for idx in questions_list:
        # print(f"==cur index is {idx}==")
        result = get_interview_question(int(idx))
        # answer = result['answer']

        # # Check if the answer length exceeds 450 characters
        # if len(answer) > 600:
        #     sentences = answer.split('。')
        #     # Remove any empty strings resulting from splitting
        #     sentences = [s for s in sentences if s]
        #     # Progressively remove the last sentence until length <= 450
        #     while sentences and len('。'.join(sentences) + '。') > 600:
        #         sentences.pop()
        #     # Reconstruct the truncated answer
        #     truncated_answer = '。'.join(sentences) + '。' if sentences else ''
        #     result['answer'] = truncated_answer

        # Check difficult level and classify
        if result['difficult'] == difficult_jd:
            matching_difficult.append(result)
        else:
            other_cases.append(result)

        # Check if we have enough matching cases
        if len(matching_difficult) >= all_value:
            break  # Stop iterating once we have enough matching cases

    # If we have enough matching cases, return the selected ones
    if len(matching_difficult) >= all_value:
        return sample(matching_difficult, all_value)
    else:
        # If not enough, take all matching cases and add from other cases to meet the required total
        remaining = all_value - len(matching_difficult)

        # Ensure remaining is valid: not greater than len(other_cases) and not negative
        remaining = min(
            remaining, len(other_cases)
        )  # Ensure remaining doesn't exceed the size of other_cases
        remaining = max(remaining, 0)  # Ensure remaining is not negative

        # Only sample from other_cases if remaining is greater than 0
        if remaining > 0:
            additional_cases = sample(other_cases, remaining)
        else:
            additional_cases = []

        return matching_difficult + additional_cases


def get_interview_question_list(label_nums: dict, difficult_jd: str):
    seed(42)
    questions_list = []

    for key, value in label_nums.items():
        interview_ids = get_interview_question_id(key)
        if len(interview_ids) >= value:

            # random_ids = random.sample(interview_ids, value) 
            random_ids = interview_ids
        else:
            print(f"Warning: Insufficient interview_ids corresponding to {key}, all are randomly selected.")
            random_ids = interview_ids
        questions_list.extend(random_ids)

    # print(f"=======Current questions_list is {questions_list}")

    if not questions_list or all(q == "" for q in questions_list):
        new_label_num = {"营销策划": 10}
        return get_interview_question_list(new_label_num, difficult_jd)

    # Make logical changes here: add the difficult index, simply and roughly extract all questions that meet the difficult_jd id, and then randomly
    question_and_answer_set = get_interview_question_with_difficulty(
        questions_list, difficult_jd, label_nums)

    return question_and_answer_set

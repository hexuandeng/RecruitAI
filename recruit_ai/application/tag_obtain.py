import concurrent.futures
import json
import re
from typing import Dict, List

from tqdm import tqdm

from recruit_ai.data_utils import sql_utils
from recruit_ai.data_utils.data_manager import PATHS, load_data_by_id
from recruit_ai.data_utils.request_model import request_model

with open(PATHS['label_map'], 'r', encoding='utf-8') as f:
    label_map = json.load(f)


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'


def is_english(char):
    return char.isalpha() and char.lower() in 'abcdefghijklmnopqrstuvwxyz'


def extract_english_letters(s):
    return ''.join([c for c in s if is_english(c)])


def extract_chinese_chars(s):
    return ''.join([c for c in s if is_chinese(c)])


def build_pattern(s):
    import re
    escaped_chars = [re.escape(c) for c in s]
    pattern = r'\s*'.join(escaped_chars)
    return pattern


def match_substring(text, substring):
    substrings = substring.split('/')

    for part in substrings:
        # Remove whitespace from part
        part = part.strip()
        if not part:
            continue

        has_english = any(is_english(c) for c in part)

        if has_english:
            # Extract English letters
            eng_part = extract_english_letters(part)
            if not eng_part:
                continue  # No English letters to match
            # Build regex pattern for eng_part
            pattern = build_pattern(eng_part)
            # Perform regex search in text
            if re.search(pattern, text, re.IGNORECASE):
                return True
        else:
            # Extract Chinese characters
            chinese_part = extract_chinese_chars(part)
            num_chars = len(chinese_part)
            # Split into first half and second half
            half = chinese_part
            if num_chars > 2:
                half = chinese_part[:2]
            pattern = build_pattern(half)
            if re.search(pattern, text):
                return True

    return False


def confirm_labels(labels: List[str], response: str) -> List[str]:
    yes_or_no = re.findall(r'是|否', response)
    result = []
    for i in range(min(len(labels), len(yes_or_no))):
        if yes_or_no[i] == '是':
            result.append(labels[i])
    return result


def tag_label(
    data: Dict[str, int | str],
    category: str,
    label_map: Dict[str, List[str]],
    task: str,
    **kwargs,
) -> List[str]:
    labels = []
    if category in label_map:
        labels += label_map[category]
    data_str = ''
    for v in data.values():
        if isinstance(v, str):
            data_str += v + '\n'
    for v in label_map.values():
        for label in v:
            if match_substring(data_str, label):
                labels.append(label)
    labels = list(set(labels))
    step = 10
    result = []

    def process_labels(label_batch: List[str]) -> List[str]:
        response = request_model(model_name='Label', task=task, labels=label_batch, **kwargs)
        return confirm_labels(label_batch, response)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(labels), step):
            label_batch = labels[i:i + step]
            futures.append(executor.submit(process_labels, label_batch.copy()))
        for future in concurrent.futures.as_completed(futures):
            result += future.result()
        print(result)
        result = process_labels(result.copy())
        print(result)
    category_cnt = {k:0 for k in label_map.keys()}
    for label in result:
        for k, v in label_map.items():
            if label in v:
                category_cnt[k] += 1
                break
    max_category = max(category_cnt, key=category_cnt.get)
    final_result =[]
    for label in result:
        if label in label_map[max_category]:
            final_result.append(label)
    return final_result


def tag_datas(
    datas: List[Dict[str, int | str]],
    data_type: str,
    kv_map: Dict[str, str],
    label_map: Dict[str, List[str]],
):
    for data in tqdm(datas):
        kwargs = {k: data[v] for k, v in kv_map.items()}
        category = request_model('Label', f'classify_{data_type}', **kwargs)
        labels = tag_label(
            data=data,
            category=category,
            label_map=label_map,
            task=f'tag_{data_type}',
            **kwargs,
        )
        print(kwargs)
        data['category'] = category
        data['labels'] = labels


def tag_obtain(types: str, id: int, update: bool):
    if not update:
        result = sql_utils.check_if_id_exists_in_tag_db(types, id)
        if result is not None:
            print(f'id {id} already exist.')
            print(f"result is {result}")
            return result[2].split(",")
    if types == 'jd':
        print('Processing JDs...')
        jds = load_data_by_id(PATHS['jd'], '需求编号', id)
        tag_datas(
            jds, 'jd', {
                'job_position': '需求岗位',
                'job_responsibility': '岗位职责',
                'job_requirement': '任职要求'
            }, label_map)
        # print(f"=== {jds}")
        print(f'tagged labels: {jds[0]["labels"]}')
        sql_utils.insert_data_into_tag_db(types, id, jds[0]['category'],
                                          ','.join(jds[0]['labels']))
        print(types, id, jds[0]['category'], ','.join(jds[0]['labels']))
        return [jds[0]['labels']][0]
    elif types == 'resume':
        print('Processing cv_works...')
        cv_works = load_data_by_id(PATHS['work'], 'pid', id)
        print(f'cv_works: {cv_works}')
        tag_datas(cv_works, 'work', {'job_description': 'description'},
                  label_map)
        print('Processing cv_projects...')
        cv_projects = load_data_by_id(PATHS['project'], 'pid', id)
        print(f'cv_project: {cv_projects}')
        tag_datas(
            cv_projects, 'project', {
                'project_description': 'project_description',
                'project_responsibility': 'project_responsibility'
            }, label_map)
        merged_datas = cv_works + cv_projects
        categorys = []
        labels = []
        for data in merged_datas:
            categorys.append(data['category'])
            labels += data['labels']
        print(f'tagged labels: {labels}')
        sql_utils.insert_data_into_tag_db(types, id, ','.join(categorys),
                                          ','.join(labels))
        print(types, id, ','.join(categorys), ','.join(labels))
        return labels
    else:
        raise ValueError('type must be jd or resume.')

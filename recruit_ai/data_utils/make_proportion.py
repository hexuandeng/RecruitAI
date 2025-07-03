import json
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from recruit_ai.data_utils.data_manager import PATHS, load_data_by_id

must_know_label = {
    "市场营销": ["市场营销"],
    "产品/项目/运营": ["产品规划", "需求分析"],
    "通信/硬件": ["通信原理", "数字电路"],
    "咨询/管理": ["管理学", "心理学"],
    "人力/财务/行政": ["人力资源管理", "财务管理", "行政管理"],
    "供应链/物流": ["供应链管理"],
    "机械/制造": ["机械设计", "机械原理", "工程力学"],
    "视觉/交互/设计": ["视觉设计", "交互设计"],
    "金融": ["金融学", "经济学", "金融风险与监管"],
    "软件开发": ["数据结构与算法", "软件工程"],
    "教育/科研": ["教育教学知识与能力"],
    "生物医药": ["病理", "药理", "细胞生物学"]
}

with open("../datas/big_small_label.json", "r", encoding="utf-8") as f:
    bs_labels = json.load(f)
all_bs_labels = []
for v in bs_labels.values():
    all_bs_labels += v
all_bs_labels = set(all_bs_labels)

def count_appearance(all_labels: List[List[List[str]]]):
    results = []
    for it in all_labels:
        count = Counter()
        for cnt in it:
            count.update(cnt)
        results.append(count)
    return results


def random_select(count: Counter, n: int):
    elements = list(count.elements())
    selected_elements = random.choices(elements, k=n)
    count.update(selected_elements)
    return dict(count)


def make_proportion(labels: List[List[Dict[str, List[str]]]],
                    total: int,
                    category_weight: List[float],
                    main_category: int = 0,
                    key_weight: Dict[str, float] = {
                        # "group": 0,
                        "label": 1
                    },
                    return_key=None):
    '''
    labels: Each element represents the label prediction result of different types of input (JD/resume/additional requirements, etc.)
    weight: The proportion of each type of input corresponding to the total number of questions
    Dict[str, List[str]]: {"group": [...], "label": [...]}
    len(labels) == len(weight)
    '''
    if return_key is not None:
        key_weight: Dict[str, float] = {
            return_key: 1
        }
    if total <= 0:
        return {}
    s = sum(category_weight)
    category_weight = [i / s for i in category_weight]
    s = sum(key_weight.values())
    key_weight = {k: v / s for k, v in key_weight.items()}

    to_process = {}

    # Count the number of occurrences of tags within each category
    for k, v in key_weight.items():
        if v == 0:
            continue
        to_process[k] = [[it[k] for it in category] for category in labels]
        to_process[k] = count_appearance(to_process[k])

    if return_key is None:
        permitted_label = list(to_process['label'][main_category].keys())
        if 'group' in to_process:
            for k in to_process['group'][main_category].keys():
                permitted_label += must_know_label[k]
        permitted_label = [i for i in permitted_label if i in all_bs_labels]
        permitted_label = set(permitted_label)

    # Allocate quotas based on category_weight and key_weight
    results = {}
    for key, count in to_process.items():
        result = defaultdict(int)
        for it, weight in zip(count, category_weight):
            cnt_weight = weight * key_weight[key]
            if it.total() > 0:
                for k, v in it.items():
                    result[k] += (v / it.total()) * cnt_weight
        results[key] = result

    # print(results)
    if return_key is not None and return_key in results:
        result = results[return_key]
        result = {k: v for k, v in result.items() if k in bs_labels.keys()}
    else:
        # Map the group to the corresponding label
        result = results["label"]
        if "group" in results:
            for k, v in results["group"].items():
                for it in must_know_label[k]:
                    result[it] += v / len(must_know_label[k])
        to_remove = []
        for k, v in result.items():
            if k not in permitted_label:
                to_remove.append(k)
        for k in to_remove:
            del result[k]

    # If too few questions are needed, one question will be asked for each label.
    max_labels = max(int(total / 1.5), 1)
    if len(result) >= max_labels:
        final = sorted(result.keys(), key=lambda x: result[x])[-max_labels:]
        result = {k: result[k] for k in final}

    # Allocate the number of questions based on total demand
    s = sum(result.values())
    used = list(result.keys()).copy()
    cnt_total = 0
    to_remove = []
    for k, v in result.items():
        try:
            label_results = load_data_by_id(
                data_path=PATHS["label_to_question"], key="label", id=k)
            label_results = label_results["question"]
            min_len = float("inf")
            for k0, v0 in label_results.items():
                min_len = min(min_len, len(v0))
            print(k, v0)
            result[k] = min(int(v / s * total), int(min_len / 3))
        except:
            result[k] = int(v / s * total)
        cnt_total += result[k]
        if result[k] <= 0:
            to_remove.append(k)
    for k in to_remove:
        del result[k]

    # The missing parts are randomly assigned to the existing labels
    if len(result) == 0:
        if len(used) == 0:
            return {}
        else:
            return random_select(Counter(used), total - len(used))
    return random_select(Counter(result), total - cnt_total)


def random_build_structure(num_inner_lists, num_dicts_per_inner_list):
    keys = list(must_know_label.keys())

    outer_list = []
    for _ in range(num_inner_lists):
        inner_list = []
        for _ in range(random.randint(0, num_dicts_per_inner_list)):
            group = random.choice(keys)
            labels = must_know_label[group]
            label_selection = random.choices(labels,
                                             k=random.randint(0, len(labels)))
            inner_dict = {"group": [group], "label": label_selection}
            inner_list.append(inner_dict)
        outer_list.append(inner_list)

    return outer_list

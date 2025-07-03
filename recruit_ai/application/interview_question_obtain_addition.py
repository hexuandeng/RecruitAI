import json
import time
from datetime import datetime
from typing import Dict, List

from recruit_ai.application.question_obtain import get_interview_question_list
from recruit_ai.application.tag_obtain import tag_obtain,tag_datas
from recruit_ai.data_utils.make_proportion import make_proportion
from recruit_ai.data_utils.data_manager import PATHS, load_data_by_id

with open(PATHS['label_map'], 'r', encoding='utf-8') as f:
    label_map = json.load(f)

all_fields = [
    "市场营销", "产品/项目/运营", "通信/硬件", "咨询/管理", "人力/财务/行政", "供应链/物流", "机械/制造",
    "视觉/交互/设计", "金融", "软件开发", "教育/科研", "生物医药"
]


def get_category_and_label(datas: str) -> List[Dict[str, List[str]]]:
    # get label
    other_datas = [{'text': datas}]
    tag_datas(other_datas, 'text', {'text': 'text'}, label_map)
    labels = other_datas[0]['labels']
    results = [{"group":[label],"label":[label]} for label in labels]
    return results


def get_interview_questions2(
        need_id: int, company_instroduction: str, project_career: str,
        interview_suggestions: str,
        interview_experience: str) -> Dict[str, List[str] | str]:
    # Loading JD data
    jd_category = tag_obtain(types="jd",id = need_id,update=False)
    if len(jd_category) == 0:
        return {'message': 'current need_id can not found jd', 'status': 1}
    # Classify and label the demand (JD) data and other input text
    start_time = time.time()

    # Difficulty rating for JD:
    jds = load_data_by_id(PATHS['jd'], '需求编号', need_id)
    if len(jds) > 0:
        duty = jds[0]["岗位职责"]
        requirement = jds[0]["任职要求"]
        jd_difficulty = jds[0]["post_level"]
    else:
        duty = None
        requirement = None
        jd_difficulty = "中级"
    jd_info = f"岗位职责：{duty}\n任职要求：{requirement}"
    
    if jd_difficulty == "初级" or jd_difficulty == "中级" or jd_difficulty == "高级":
        difficulty = jd_difficulty
    else:
        difficulty = "中级"

    jd_category_label = [{"group":[group],"label":[group]} for group in jd_category]
    interview_suggestions_category_label = get_category_and_label(interview_suggestions)
    interview_experience_category_label = get_category_and_label(interview_experience)
    end_time = time.time()
    print(f"model generate time is {end_time - start_time}")
    # Calculate weights based on the obtained categories and label generation results
    all_types_category_label = [
        interview_suggestions_category_label,
        interview_experience_category_label, jd_category_label
    ]
    label_nums = make_proportion(labels=all_types_category_label,
                                 total=10,
                                 category_weight=[0.5, 0.5, 0.5],
                                 key_weight={
                                     "group": 0,
                                     "label": 1.0
                                 })
    print(f"label_num is {label_nums}")

    # Index the topic based on the obtained label and corresponding weight
    # Loading interview question data (stored by major categories)
    question_and_answer_set = get_interview_question_list(label_nums,difficulty)
    # Construct the required return structure
    result = {
        "message": "success",
        "result": {
            "question_and_answer_set": []
        },
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": 0
    }
    for question_and_answer in question_and_answer_set:
        result["result"]["question_and_answer_set"].append({
            "question":
            question_and_answer["question"],
            "answer":
            question_and_answer["answer"]
        })
    return result

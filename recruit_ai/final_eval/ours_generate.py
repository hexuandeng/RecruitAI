import argparse
import json
import logging
import time

from recruit_ai.application.one_point_three_offline import \
    get_interview_questions
from recruit_ai.application.question_obtain import question_obtain
from recruit_ai.data_utils.data_manager import PATHS, load_data_by_id
from tqdm import tqdm

# Configuring Logging
logging.basicConfig(
    filename='Qwen2-7B-Instruct-new_all_performance.log',  # Output log file name
    level=logging.INFO,  # Log Level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log Format
)


def format_questions(question_list):
    formatted_output = []
    question_number = 1
    for question in question_list:
        formatted_output.append(f"题目 {question_number}：{question['question']}")
        if 'options' in question:
            formatted_output.append(
                f"选项 {question_number}：{', '.join(question['options'])}")
        formatted_output.append(f"答案 {question_number}：{question['answer']}")
        question_number += 1
    return "\n".join(formatted_output)


def save_data_mode1(data, data_file):
    with open(data_file, "w", encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def save_data_mode2(data, data_file):
    with open(data_file, "w", encoding="utf-8") as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


def load_data_mode1(data_file):
    data = json.load(open(data_file, "r", encoding='utf-8'))
    return data


def load_data(cv_id, need_idset):
    jds = []
    for need_id in need_idset:
        jd = load_data_by_id(PATHS['jd'], '需求编号', need_id)
        jds.extend(jd)
    cv_works = load_data_by_id(PATHS['work'], 'pid', cv_id)
    cv_projects = load_data_by_id(PATHS['project'], 'pid', cv_id)

    print(f"cur_jds is {jds}")

    if len(jds) > 0:
        duty = "\n\n".join(item["岗位职责"] for item in jds
                           if item.get("岗位职责") is not None)
        requirement = "\n\n".join(item["任职要求"] for item in jds
                                  if item.get("任职要求") is not None)
    else:
        duty = None
        requirement = None

    if len(cv_works) > 0:
        job_description = "\n\n".join(
            item["job_description"] for item in cv_works
            if item.get("job_description") is not None)
    else:
        job_description = None
    if len(cv_projects) > 0:
        # Concatenate all projects' project_description
        project_description = "\n\n".join(
            item["project_description"] for item in cv_projects
            if item.get("project_description") is not None)

        # Splice all projects' project_responsibility
        project_duty = "\n\n".join(
            item["project_responsibility"] for item in cv_projects
            if item.get("project_responsibility") is not None)
    else:
        project_description = None
        project_duty = None

    jd_info = f"岗位职责：{duty}\n任职要求：{requirement}"
    resume_info = f"工作描述：{job_description}\n项目描述：{project_description}\n项目职责：{project_duty}"

    return jd_info, resume_info


def main(args: argparse.Namespace):
    print("Generating written test questions...")
    with open(args.exam_test_case, 'r', encoding='utf-8') as f:
        exam_datas = json.load(f)[:1000]
    exam_result = []
    question_obtain_total_time = 0
    question_obtain_count = 0
    for exam_data in tqdm(exam_datas):
        cv_id = exam_data["cv_id"]
        need_idset = [int(x) for x in exam_data["need_idset"].split(",")]
        jd_info, resume_info = load_data(cv_id, need_idset)
        start_time = time.time()
        our_result = question_obtain(**exam_data)
        print("======================our result======================")
        print(our_result)
        end_time = time.time()
        our_time = end_time - start_time
        question_obtain_total_time += our_time
        question_obtain_count += 1
        exam_result.append({
            "cv_id": cv_id,
            "need_id": need_idset,
            "our_response_time": our_time,
            "base_response_time": None,
            "jd_info": jd_info,
            "resume_info": resume_info,
            "our": {
                "fill_in_question":
                our_result['result']['question_and_answer_set']
                ['fill_in_questions'] if 'result' in our_result else None,
                "choice_question":
                our_result['result']['question_and_answer_set']
                ['choice_questions'] if 'result' in our_result else None,
                "true_or_false_question":
                our_result['result']['question_and_answer_set']
                ['true_or_false_questions']
                if 'result' in our_result else None,
            },
            "baseline": None
        })

    question_obtain_avg_time = question_obtain_total_time / question_obtain_count if question_obtain_count > 0 else 0

    logging.info(
        f"Average time for exam question_obtain: {question_obtain_avg_time:.2f}s"
    )
    print(
        f"Average time for exam question_obtain: {question_obtain_avg_time:.2f}s"
    )
    with open(args.exam_output_path, 'w', encoding='utf-8') as f:
        json.dump(exam_result, f, ensure_ascii=False, indent=4)

    print("Generating interview questions...")
    with open(args.interview_test_case, 'r', encoding='utf-8') as f:
        interview_datas = json.load(f)[:1000]
    interview_result = []
    question_obtain_total_time = 0
    question_obtain_count = 0
    for interview_data in tqdm(interview_datas):
        cv_id = interview_data["cv_id"]
        need_id = interview_data["need_id"]
        print(f"current cv_id is{cv_id}, need_id is {need_id}")
        jd_info, resume_info = load_data(cv_id, [need_id])
        start_time = time.time()
        our_result = get_interview_questions(cv_id, need_id)
        end_time = time.time()
        our_time = end_time - start_time
        question_obtain_total_time += our_time
        question_obtain_count += 1
        interview_result.append({
            "cv_id":
            cv_id,
            "need_id": [need_id],
            "our_response_time":
            our_time,
            "base_response_time":
            None,
            "jd_info":
            jd_info,
            "resume_info":
            resume_info,
            "our":
            our_result['result']['question_and_answer_set']
            if 'result' in our_result else None,
            "baseline":
            None
        })
    question_obtain_avg_time = question_obtain_total_time / question_obtain_count if question_obtain_count > 0 else 0
    logging.info(
        f"Average time for interview question_obtain: {question_obtain_avg_time:.2f}s"
    )
    print(
        f"Average time for interview question_obtain: {question_obtain_avg_time:.2f}s"
    )
    with open(args.interview_output_path, 'w', encoding='utf-8') as f:
        json.dump(interview_result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam_test_case",
                        type=str,
                        default="exam_test_case.json")
    parser.add_argument("--interview_test_case",
                        type=str,
                        default="interview_test_case.json")
    parser.add_argument("--exam_output_path",
                        type=str,
                        default="eval_pipeline/our_exam.json")
    parser.add_argument("--interview_output_path",
                        type=str,
                        default="eval_pipeline/our_interview.json")
    main(parser.parse_args())

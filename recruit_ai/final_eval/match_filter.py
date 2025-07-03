import argparse
import json
import re

import torch
import vllm
from tqdm import tqdm

interview_PIPEI_PROMPT = """作为面试评估专家，请评估以下面试题目与JD的匹配程度。

岗位JD信息：
【任职要求】
{job_requirements}

【岗位职责】
{job_responsibilities}

面试题目：
{interview_questions}

请先给出匹配度分析，然后给出匹配度评分，分数范围在1-10分，高匹配度题目（8-10分），中等匹配度题目（4-7分），低匹配度题目（1-3分）。

注意：请在评估的末尾，给出最终评分，以下述格式给出：
**分析**：[分析]
**最终分数**：[分数]
"""


def extrac_job_info(job_infos):
    job_reqs = []
    job_res = []
    for jd_info in job_infos:
        try:
            # Preprocessing JD information
            # Use regular expressions to extract HTML content of job responsibilities and job requirements
            job_responsibilities_match = re.search(r'岗位职责：<p>(.*?)</p>',
                                                   jd_info, re.DOTALL)
            job_requirements_match = re.search(r'任职要求：<p>(.*?)</p>', jd_info,
                                               re.DOTALL)

            # If the corresponding content is found, further processing is performed
            if job_responsibilities_match:
                job_responsibilities_html = job_responsibilities_match.group(1)
                # Remove HTML tags and replace with line breaks
                job_responsibilities = re.sub(r'<[^>]+>', '\n',
                                              job_responsibilities_html)
            else:
                job_responsibilities = ""
            if job_requirements_match:
                job_requirements_html = job_requirements_match.group(1)
                # Remove HTML tags and replace with line breaks
                job_requirements = re.sub(r'<[^>]+>', '\n',
                                          job_requirements_html)
            else:
                job_requirements = ""
            job_reqs.append(job_requirements)
            job_res.append(job_responsibilities)
        except Exception as e:
            print(f"Error extracting resume information: {str(e)}")
            job_responsibilities = ""
            job_requirements = ""
            job_reqs.append(job_requirements)
            job_res.append(job_responsibilities)
            return ""  #If the overall evaluation fails, an empty string is also returned
    return job_reqs, job_res


def question_format(item):
    question = f'题目: {item["question"] if "question" in item else None}\n答案: {item["answer"] if "answer" in item else None}'
    return question


def question_format_list(data):
    q_lsit = []
    if data is None:
        return [""] * 20
    for item in data:
        question = f'题目: {item["question"] if "question" in item else None}\n答案: {item["answer"] if "answer" in item else None}'
        q_lsit.append(question)
    return q_lsit


def generate(model, prompts):

    # setting generate args
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=2048,
        stop=["\n\n\n\n"],
    )

    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {g.prompt: g.outputs[0].text for g in generations}
    outputs = [
        prompt_to_output[prompt] if prompt in prompt_to_output else ""
        for prompt in prompts
    ]

    return outputs


def pipei_eval(job_requirements, job_responsibilities, question_list, model):
    all_prompts = []
    batch_counts = []
    # Collect all prompts and record the number of questions for each position
    for job_req, job_res, question_l in tqdm(zip(job_requirements,
                                                 job_responsibilities,
                                                 question_list),
                                             desc="Generate prompt batch"):
        prompts = []
        for question in question_l:
            try:
                prompt = interview_PIPEI_PROMPT.format(
                    job_requirements=job_req,
                    job_responsibilities=job_res,
                    interview_questions=question)
                prompts.append(prompt)
            except Exception as e:
                print(f"Generate prompt error: {str(e)}")
                prompts.append("")
        batch_counts.append(len(prompts))
        all_prompts.extend(prompts)

    # Batch reasoning for all prompts
    print("Start batch inference...")
    all_outputs = generate(model, all_prompts)

    # Split the results by original batch
    eval_results = []
    current_idx = 0
    for count in batch_counts:
        sub_eval = all_outputs[current_idx:current_idx + count]
        eval_results.append(sub_eval)
        current_idx += count

    return eval_results


def parse_match_score(evaluation_text):
    if evaluation_text is None:
        return -1

    def extract_last_number(text):
        # Use regular expressions to match numbers (integers or decimals)
        numbers = re.findall(r'\d+\.\d+|\d+', text)

        # Returns the last number in the list (if any)
        if numbers:
            return numbers[-1]
        else:
            return -1

    try:
        # Try multiple regular expression matching patterns
        difficulty_patterns = [
            # Original matching mode
            r'\*\*最终分数\*\*[：:]\s*(\d+\.?\d*)分?',
            # 匹配 **最终分数**：**4.27** 格式
            r'\*\*最终分数\*\*[：:]\s*\*\*(\d+\.?\d*)\*\*',
            r'最终分数[\""\*]*：\s*(\d+\.?\d*)'
        ]
        difficulty_score = -1
        for pattern in difficulty_patterns:
            score_match = re.search(pattern, evaluation_text, re.DOTALL)
            if score_match:
                difficulty_score = float(score_match.group(1))
                break
        if difficulty_score == -1:
            difficulty_score = extract_last_number(evaluation_text)
            difficulty_score = float(
                difficulty_score) if difficulty_score != -1 else -1

        return difficulty_score

    except Exception as e:
        print(f"Error parsing evaluation results: {str(e)}")
        print(f"Error Type: {type(e)}")
        # print(f"Original text: {evaluation_text[:200]}...")
        return -1


def filt(pipei_eval_results, eval_data, exam_key=None):
    pipei_score = [[parse_match_score(item) for item in sublist]
                   for sublist in pipei_eval_results]
    print(pipei_score)
    # Filter and select the top 10 matches with the highest degree
    assert len(pipei_score) == len(eval_data)
    for scores, data in zip(pipei_score, eval_data):
        if data is None or "baseline" not in data or data["baseline"] is None:
            # If data is None or does not contain the "baseline" key, this loop is skipped.
            continue
        if exam_key is not None and exam_key not in data["baseline"]:
            continue
        if exam_key is None:
            item = data["baseline"]
            topk = 10
        else:
            item = data["baseline"][exam_key]
            topk = 5
        combined = list(zip(scores, item))
        combined.sort(reverse=True, key=lambda x: x[0])
        top_k_combined = combined[:topk]
        top_ques = [x[1] for x in top_k_combined]
        if exam_key is None:
            data["baseline"] = top_ques
        else:
            data["baseline"][exam_key] = top_ques

    if exam_key is None:
        for idx, data in enumerate(eval_data):
            data["match_eval_results"] = pipei_eval_results[idx]
            data["match_score"] = pipei_score[idx]
    else:
        for idx, data in enumerate(eval_data):
            if "match_eval_results" not in data:
                data["match_eval_results"] = {}
            data["match_eval_results"][exam_key] = pipei_eval_results[idx]
            if "match_score" not in data:
                data["match_score"] = {}
            data["match_score"][exam_key] = pipei_score[idx]


def main(args):
    # load model
    model = vllm.LLM(
        model=args.model_path,
        #  tensor_parallel_size=torch.cuda.device_count(),
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.95,
        max_model_len=4096)
    # File Reading
    print("File Reading")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    if args.debug:
        eval_data = eval_data[:5]
    print("Pre-extraction")
    # Combined Data
    jd_infos = [item['jd_info'] for item in eval_data]
    print("Conduct a match assessment")
    if args.judge_type == 'interview':
        question_list = [
            question_format_list(item["baseline"]) if item["baseline"] else ""
            for item in eval_data
        ]
        job_requirements, job_responsibilities = extrac_job_info(jd_infos)

        pipei_eval_results = pipei_eval(job_requirements, job_responsibilities,
                                        question_list, model)

        filt(pipei_eval_results, eval_data)
    elif args.judge_type == 'exam':
        for exam_key in [
                'fill_in_question', 'choice_question', 'true_or_false_question'
        ]:
            question_list = [
                question_format_list(item["baseline"][exam_key])
                if item["baseline"] is not None else [] for item in eval_data
            ]
            job_requirements, job_responsibilities = extrac_job_info(jd_infos)
            pipei_eval_results = pipei_eval(job_requirements,
                                            job_responsibilities,
                                            question_list, model)
            filt(pipei_eval_results, eval_data, exam_key)
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default='/data/shared/Qwen2.5-7B-Instruct')
    parser.add_argument(
        "--input_file",
        type=str,
        default=
        "qwen2.5-7b-instruct-1.3.json",
        help="The file to be evaluated contains the results of baseline and ours")
    parser.add_argument('--judge_type', type=str, choices=['exam', 'interview'])
    parser.add_argument("--output_file",
                        type=str,
                        default="./pipei_filter.jsonl",
                        help="Evaluation result saving path")
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())

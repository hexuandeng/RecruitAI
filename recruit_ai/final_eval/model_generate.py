import argparse
import json
import time

import torch
from recruit_ai.data_utils.data_manager import PATHS, load_data_by_id
from vllm import LLM, SamplingParams

# The template definition remains unchanged
choice_template = [{
    "question":
    "执行以下程序，如果希望输出 df 对象中 x 列数据大于其平均值的所有行，下列选项中，做法正确的一项是（）",
    "options": [
        "print(df.query('x > a'))", "print(df.query('x > @a'))",
        "print(df.query('x > &a'))", "print(df.query('x > $a'))"
    ],
    "answer":
    "B",
}, {
    "question":
    "Linux操作系统的系统用户密码存储在哪个文件？",
    "options": ["/etc/shadow", "/etc/passwd", "/etc/group", "/etc/profile"],
    "answer":
    "A",
}]

true_fill_in_template = [{
    "question": "____是一种用于管理文件和目录的树状结构。",
    "answer": "目录树",
}, {
    "question": "____是一种用于支持多线程程序设计的操作系统模型。",
    "answer": "线程模型",
}]
wrong_fill_in_template = [{
    "question": "以下哪个不是Spring Cloud组件？____",
    "answer": 'Hibernate'
}, {
    "question": "在JavaScript中，以下哪个函数可以用来检测一个对象是否包含某个特定的键？____( )____",
    "answer": "hasOwnProperty"
}, {
    "question":
    "对于高并发场景下的线程安全，以下哪种方式不适合解决（ ）。\nA. 同步方法\nB. 同步块\nC. 使用 volatile 关键字\nD. 线程池",
    "answer": "D"
}]

true_or_false_template = [{
    "question": "在操作系统中，加载程序(loader)的作用是为程序分配内存空间并解析对象迭卡之间的符号引用。",
    "answer": "是",
}, {
    "question": "shutdown -r -t 60 是Windows定时1分钟后重启电脑的命令。",
    "answer": "是",
}]

interview_template = [{
    "question":
    "请简述你对游戏测试工具开发的理解，并举例说明你如何提升游戏质量和效率？",
    "answer":
    "游戏测试工具开发旨在通过自动化手段提高测试效率和覆盖度，减少人为错误，提升游戏质量。例如，通过开发自动化测试脚本，可以快速执行大量重复性测试，确保游戏在不同环境和条件下稳定运行。同时，利用性能测试工具可以模拟高并发用户场景，检测游戏在极限情况下的表现，从而优化游戏性能。"
}, {
    "question":
    "请描述你使用过的主流web后端框架，并简述其特点和优势。",
    "answer":
    "我熟悉Django框架，它是一个高级的Python Web框架，强调代码的可读性和可维护性。Django内置了ORM（对象关系映射）、认证系统、表单处理、URL路由等功能，使得开发过程更加高效。其优势在于快速开发、安全、可扩展性好，适合构建功能丰富的Web应用。"
}]


def get_post_level(need_id):
    print("============jd===========")
    jd = load_data_by_id(PATHS['jd'], '需求编号', need_id)
    print(jd)
    if jd[0]["post_level"] in ["初级", "中级", "高级"]:
        post_level = jd[0]["post_level"]
    else:
        post_level = "中级"
    return post_level


def load_data(cv_id, need_idset):
    jds = []
    for need_id in need_idset:
        jd = load_data_by_id(PATHS['jd'], '需求编号', need_id)
        jds.extend(jd)
    cv_works = load_data_by_id(PATHS['work'], 'pid', cv_id)
    cv_projects = load_data_by_id(PATHS['project'], 'pid', cv_id)

    if len(jds) > 0:
        duty = "\n\n".join(item["岗位职责"] for item in jds
                           if item.get("岗位职责") is not None)
        requirement = "\n\n".join(item["任职要求"] for item in jds
                                  if item.get("任职要求") is not None)
        require_job = "、".join(item["需求岗位"] for item in jds
                               if item.get("需求岗位") is not None)

    else:
        duty = None
        requirement = None
        require_job = ""

    if len(cv_works) > 0:
        job_description = "\n\n".join(
            item["job_description"] for item in cv_works
            if item.get("job_description") is not None)
    else:
        job_description = None
    if len(cv_projects) > 0:
        project_description = "\n\n".join(
            item["project_description"] for item in cv_projects
            if item.get("project_description") is not None)
        project_duty = "\n\n".join(
            item["project_responsibility"] for item in cv_projects
            if item.get("project_responsibility") is not None)
    else:
        project_description = None
        project_duty = None

    jd_info = f"岗位职责：{duty}\n任职要求：{requirement}"
    resume_info = f"工作描述：{job_description}\n项目描述：{project_description}\n项目职责：{project_duty}"

    return jd_info, resume_info, require_job


def generate_prompts_for_exam(
    exam_datas,
    choice_num: int,
    fill_in_num: int,
    true_or_false_num: int,
):
    prompts = []
    context = []
    for exam_data in exam_datas:
        cv_id = exam_data["cv_id"]
        need_idset = [int(x) for x in exam_data["need_idset"].split(",")]
        jd_info, resume_info, require_job = load_data(cv_id, need_idset)

        # Generate multiple choice prompts
        choice_prompt = (
            f"你是一个招聘技术笔试题生成助手，请你仔细阅读招聘岗位描述：{jd_info}。\n"
            f"然后根据岗位描述，针对该岗位生成{choice_num}道选择题和对应答案作为技术笔试题考察候选人，注意：生成的选择题是给候选人作答考察其技术能力。\n"
            f"输出为json格式，输出格式参考：{json.dumps(choice_template, ensure_ascii=False)}\n"
            f"注意1：请严格按照参考格式进行输出！每道选择题应包括问题、四个选项（A、B、C、D）以及正确答案。\n"
            f"注意2：只要输出json格式，其中所有属性名、字符串都必须用双引号括起来，不要多余文字。"
            f"注意3：必须使用 ```json ...``` 包裹完整的JSON输出，不要输出任何额外解释或文字。")
        prompts.append(choice_prompt)
        context.append({
            "type": "choice",
            "cv_id": cv_id,
            "需求岗位": require_job,
            "need_idset": need_idset,
            "jd_info": jd_info,
            "resume_info": resume_info
        })

        # Generate fill-in-the-blank prompts
        fill_in_prompt = (
            f"你是一个招聘技术笔试题生成助手，请你仔细阅读招聘岗位描述：{jd_info}。\n"
            f"然后根据岗位描述，针对该岗位生成{fill_in_num}道填空题和对应答案作为技术笔试题考察候选人，注意：填空题要有挖空并且生成的填空题是给候选人作答考察其技术能力。\n"
            f"输出json格式，输出格式参考：{json.dumps(true_fill_in_template, ensure_ascii=False)}\n"
            f"尤其注意：填空题的问题必须是挖空的陈述文本以供候选人填空，不能是选择题或者问答！以下是一些错误输出案例，请你避免：{json.dumps(wrong_fill_in_template, ensure_ascii=False)}\n"
            f"注意1：请严格按照参考格式进行输出！每道填空题包括问题和对应答案。\n"
            f"注意2：只要输出json格式，其中所有属性名、字符串都必须用双引号括起来，不要多余文字。"
            f"注意3：必须使用 ```json ...``` 包裹完整的JSON输出，不要输出任何额外解释或文字。")
        prompts.append(fill_in_prompt)
        context.append({
            "type": "fill_in",
            "cv_id": cv_id,
            "需求岗位": require_job,
            "need_idset": need_idset,
            "jd_info": jd_info,
            "resume_info": resume_info
        })

        # Generate True or False Question Prompt
        true_false_prompt = (
            f"你是一个招聘技术笔试题生成助手，请你仔细阅读招聘岗位描述：{jd_info}。\n"
            f"然后根据岗位描述，针对该岗位生成{true_or_false_num}道判断题和对应答案作为技术笔试题考察候选人，注意：生成的判断题是给候选人作答考察其技术能力。\n"
            f"输出为json格式，输出格式参考：{json.dumps(true_or_false_template, ensure_ascii=False)}\n"
            f"注意1：请严格按照参考格式进行输出！每道判断题包括问题和对应答案。\n"
            f"注意2：只要输出json格式，其中所有属性名、字符串都必须用双引号括起来，不要多余文字。"
            f"注意3：必须使用 ```json ...``` 包裹完整的JSON输出，不要输出任何额外解释或文字。")
        prompts.append(true_false_prompt)
        context.append({
            "type": "true_false",
            "cv_id": cv_id,
            "need_idset": need_idset,
            "需求岗位": require_job,
            "jd_info": jd_info,
            "resume_info": resume_info
        })

    return prompts, context


def generate_prompts_for_interview(interview_datas, interview_num: int):
    prompts = []
    context = []
    for interview_data in interview_datas:
        cv_id = interview_data["cv_id"]
        need_id = interview_data["need_id"]
        jd_info, resume_info, require_job = load_data(cv_id, [need_id])
        # Generate interview question prompts
        interview_prompt = (
            f"你是一位资深面试官，请你仔细阅读招聘岗位描述：{jd_info}。\n"
            f"然后根据岗位描述，针对该岗位生成{interview_num}道面试简答题和对应答案，注意：需要考察应试者对岗位相关知识的理解。\n"
            f"输出为json格式，输出格式参考：{json.dumps(interview_template, ensure_ascii=False)}\n"
            f"只要输出json格式，其中所有属性名、字符串都必须用双引号括起来，不要多余文字。"
            f"注意3：必须使用 ```json ...``` 包裹完整的JSON输出，不要输出任何额外解释或文字。")
        prompts.append(interview_prompt)
        context.append({
            "cv_id": cv_id,
            "need_id": need_id,
            "需求岗位": require_job,
            "jd_info": jd_info,
            "resume_info": resume_info
        })
    return prompts, context


def generate(prompts: list[str]) -> list[str]:
    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {
        g.prompt: g.outputs[0].text.strip()
        for g in generations
    }
    return [prompt_to_output.get(prompt, "") for prompt in prompts]


def main(args: argparse.Namespace):
    global model, sampling_params
    model = LLM(
        model=args.model_path,
        trust_remote_code=True,
        # tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        max_model_len=8192,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        presence_penalty=1.1,
        top_p=0.95,
        max_tokens=4096,
        stop_token_ids=[model.get_tokenizer().eos_token_id],
    )

    with open(args.exam_case, 'r', encoding='utf-8') as f:
        exam_datas = json.load(f)[:1000]
    if args.debug:
        exam_datas = exam_datas[:1]
    exam_prompts, exam_context = generate_prompts_for_exam(
        exam_datas,
        args.choice_num,
        args.fill_in_num,
        args.true_or_false_num,
    )
    exam_start_time = time.time()
    exam_generated = generate(exam_prompts)
    exam_end_time = time.time()
    avg_exam_time = (exam_end_time - exam_start_time) / len(exam_datas)
    exam_result = []
    # Because each exam_data generates 3 questions, they are processed in groups of 3.
    print("Generate written test questions...")
    for idx in range(0, len(exam_context), 3):
        exam_data = exam_context[idx]
        require_job = exam_data["需求岗位"]
        cv_id = exam_data["cv_id"]
        need_idset = exam_data["need_idset"]
        jd_info = exam_data["jd_info"]
        resume_info = exam_data["resume_info"]
        post_level = get_post_level(need_idset[0])
        try:
            # Each exam_data has 3 prompts: choice, fill_in, true_false
            choice_output = exam_generated[idx]
            fill_in_output = exam_generated[idx + 1]
            true_false_output = exam_generated[idx + 2]
            try:
                start = "```json"
                end = "```"
                start_index = choice_output.index(start) + len(start)
                end_index = choice_output.index(end, start_index)
                choice_output = choice_output[start_index:end_index]
                # print(f"Generated multiple choice questions JSON: {choice_output.strip()}")
                # print("===================")
                # print(f"current start is {start_index}, end is {end_index}")
                # print("===================")
                start_index = fill_in_output.index(start) + len(start)
                end_index = fill_in_output.index(end, start_index)
                fill_in_output = fill_in_output[start_index:end_index]

                start_index = true_false_output.index(start) + len(start)
                end_index = true_false_output.index(end, start_index)
                true_false_output = true_false_output[start_index:end_index]

            except Exception as e:
                # Debug: Print generated output
                print(f"Generated multiple choice questions JSON: {choice_output}")
                print(f"Generated fill-in-the-blank questions JSON: {fill_in_output}")
                print(f"Generated true or false questions JSON: {true_false_output}")
                print(e)

            # Parsing JSON
            choice_questions = json.loads(choice_output.strip())
            fill_in_questions = json.loads(fill_in_output.strip())
            true_or_false_questions = json.loads(true_false_output.strip())
            print("============success===============")

            question_and_answer_set = {
                "fill_in_question": fill_in_questions,
                "choice_question": choice_questions,
                "true_or_false_question": true_or_false_questions,
            }

            exam_result.append({
                "cv_id": cv_id,
                "需求岗位": require_job,
                "need_id": need_idset,
                "jd_info": jd_info,
                "our_response_time": None,
                "base_response_time": avg_exam_time,
                "resume_info": resume_info,
                "our": None,
                "baseline": question_and_answer_set,
                "post_level": post_level
            })
        except json.JSONDecodeError as e:
            print(f"Parsing Error: {e}")
            print(f"Generated multiple choice output: {choice_output}")
            print(f"Generated fill-in-the-blank output: {fill_in_output}")
            print(f"Generated true or false question output: {true_false_output}")
            # Parsing errors are also added:
            exam_result.append({
                "cv_id": cv_id,
                "need_id": need_idset,
                "需求岗位": require_job,
                "our_response_time": None,
                "base_response_time": avg_exam_time,
                "jd_info": jd_info,
                "resume_info": resume_info,
                "our": None,
                "baseline": None,
                "post_level": post_level
            })
            continue
        except IndexError as e:
            print(f"Index Error: {e}")
            continue
    with open(args.exam_output, "w", encoding='utf-8') as f:
        json.dump(exam_result, f, ensure_ascii=False, indent=4)

    with open(args.interview_case, 'r', encoding='utf-8') as f:
        interview_datas = json.load(f)[:1000]
    if args.debug:
        interview_datas = interview_datas[:1]
    interview_prompts, interview_context = generate_prompts_for_interview(
        interview_datas,
        args.interview_num,
    )
    print(
        f"Interview prompt length：{len(interview_prompts)}，Interview context length：{len(interview_context)}"
    )
    interview_start_time = time.time()
    interview_generated = generate(interview_prompts)
    interview_end_time = time.time()
    interview_avg_time = (interview_end_time -
                        interview_start_time) / len(interview_datas)
    interview_result = []
    print("Generate interview questions...")
    for idx, interview_data in enumerate(interview_context):
        cv_id = interview_data["cv_id"]
        require_job = interview_data["需求岗位"]
        need_id = interview_data["need_id"]
        jd_info = interview_data["jd_info"]
        resume_info = interview_data["resume_info"]
        post_level = get_post_level(need_id)
        try:
            # Debug: Print generated output
            interview_output = interview_generated[idx]
            # print(f"Generated interview questions JSON: {interview_output}")
            try:
                start = "```json"
                end = "```"
                # Find the start and end positions and add exception capture
                try:
                    start_index = interview_output.index(start) + len(start)
                    end_index = interview_output.index(end, start_index)
                except ValueError as e:
                    print(
                        f"Error: start or end identifier not found, current interview_output content：{interview_output}"
                    )
                    interview_result.append({
                        "cv_id": cv_id,
                        "need_id": [need_id],
                        "our_response_time": None,
                        "需求岗位": require_job,
                        "base_response_time": interview_avg_time,
                        "jd_info": jd_info,
                        "resume_info": resume_info,
                        "our": None,
                        "baseline": None,
                        "post_level": post_level
                    })
                    continue  # If start or end is not found, the current item is skipped.
                # print("===================")
                # print(f"current start is {start_index}, end is {end_index}")
                # print("===================")
                interview_output = interview_output[start_index:end_index]
                # print("===================success====================")
            except Exception as e:
                # Debug: Print generated output
                # print(f"Generated multiple choice questions JSON: {interview_output}")
                print(e)
                interview_result.append({
                    "cv_id": cv_id,
                    "need_id": [need_id],
                    "our_response_time": None,
                    "需求岗位": require_job,
                    "base_response_time": interview_avg_time,
                    "jd_info": jd_info,
                    "resume_info": resume_info,
                    "our": None,
                    "baseline": None,
                    "post_level": post_level
                })
                break
            # Parsing JSON
            question_and_answer_set = json.loads(interview_output)

            interview_result.append({
                "cv_id": cv_id,
                "need_id": [need_id],
                "our_response_time": None,
                "base_response_time": interview_avg_time,
                "需求岗位": require_job,
                "jd_info": jd_info,
                "resume_info": resume_info,
                "our": None,
                "baseline": question_and_answer_set,
                "post_level": post_level
            })
        except json.JSONDecodeError as e:
            print(f"Parsing Error: {e}")
            #print(f"Generated interview question output: {interview_output}")
            # Error generation must also be added, and then compared
            interview_result.append({
                "cv_id": cv_id,
                "need_id": [need_id],
                "our_response_time": None,
                "需求岗位": require_job,
                "base_response_time": interview_avg_time,
                "jd_info": jd_info,
                "resume_info": resume_info,
                "our": None,
                "baseline": None,
                "post_level": post_level
            })
            continue
        except IndexError as e:
            print(f"Index Error: {e}")
    with open(args.interview_output, "w", encoding='utf-8') as f:
        json.dump(interview_result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=
        "Qwen2.5-7B-Instruct",
        help='Local model path')
    parser.add_argument('--exam_case',
                        type=str,
                        default="exam_test_case.json")
    parser.add_argument('--interview_case',
                        type=str,
                        default="interview_test_case.json")
    parser.add_argument('--exam_output', type=str, default='glm-9b1.1.json')
    parser.add_argument('--interview_output', type=str, default='glm-9b1.3.json')
    parser.add_argument('--temperature', type=float, default=0.1, help='Generation temperature')
    parser.add_argument('--choice_num', type=int, default=5, help='Number of multiple choice questions')
    parser.add_argument('--fill_in_num', type=int, default=5, help='Number of fill-in-the-blank questions')
    parser.add_argument('--true_or_false_num',
                        type=int,
                        default=5,
                        help='Number of True or False Questions')
    parser.add_argument('--interview_num', type=int, default=10, help='Number of interview questions')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)

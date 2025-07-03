import argparse
import concurrent.futures
import json
import re
import time

from openai import OpenAI
from tqdm import tqdm

MODEL_TYPE = "deepseek-v3-250324"
API_KEY = ""
API_BASE = ""

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

MAX_TRY = 20000

exam_PIPEI_PROMPT = """作为面试评估专家，请评估以下面试题目与JD的匹配程度。

岗位JD信息：
【任职要求】
{job_requirements}

【岗位职责】
{job_responsibilities}

面试题目（总计15道题）：
【选择题】
{choice_questions}

【填空题】
{fill_in_questions}

【判断题】
{true_or_false_questions}

请从以下方面详细评估题目与JD的匹配度：

1. 单个题目匹配度评分
请对每道题目的JD匹配度进行评分（1-10分），并说明理由：
   选择题1: [分数] - [理由：对应JD中哪些具体要求]
   选择题2: [分数] - [理由：对应JD中哪些具体要求]
   ...
   填空题1: [分数] - [理由：对应JD中哪些具体要求]
   ...
   判断题1: [分数] - [理由：对应JD中哪些具体要求]
   ...

2. 总体匹配度评分（1-10分）
   - 计算方法：所有题目的匹配度评分的平均值
   - 给出计算过程和最终分数

3. 匹配分析
   高匹配度题目（8-10分）：
   - 列举题目编号及对应的JD技能点
   
   中等匹配度题目（4-7分）：
   - 列举题目编号及对应的JD技能点
   
   低匹配度题目（1-3分）：
   - 列举题目编号
   - 说明不匹配原因

4. 改进建议
   a) 未覆盖的重要JD技能点
   b) 低匹配度题目的具体改进方案
   c) 建议补充的题目类型和内容

请在评估的末尾，给出最终评分，以下述格式给出：
**最终分数**：[分数]
"""

interview_PIPEI_PROMPT = """作为面试评估专家，请评估以下面试题目与JD的匹配程度。

岗位JD信息：
【任职要求】
{job_requirements}

【岗位职责】
{job_responsibilities}

面试题目：
{interview_questions}

请从以下方面详细评估题目与JD的匹配度：

1. 单个题目匹配度评分
请对每道题目的JD匹配度进行评分（1-10分），并说明理由：
   题目1: [分数] - [理由：对应JD中哪些具体要求]
   题目2: [分数] - [理由：对应JD中哪些具体要求]
   ...

2. 总体匹配度评分（1-10分）
   - 计算方法：所有题目的匹配度评分的平均值
   - 给出计算过程和最终分数

3. 匹配分析
   高匹配度题目（8-10分）：
   - 列举题目编号及对应的JD技能点
   
   中等匹配度题目（4-7分）：
   - 列举题目编号及对应的JD技能点
   
   低匹配度题目（1-3分）：
   - 列举题目编号
   - 说明不匹配原因

4. 改进建议
   a) 未覆盖的重要JD技能点
   b) 低匹配度题目的具体改进方案
   c) 建议补充的题目类型和内容

请在评估的末尾，给出最终评分，以下述格式给出：
**最终分数**：[分数]
"""

exam_DIFFICULT_PROMPT = """作为专业面试评估专家，请全面评估以下面试题目的难度。

岗位JD信息：
【任职要求】
{job_requirements}

【岗位职责】
{job_responsibilities}

面试题目（总计{questions_num}道）：

【选择题】
{choice_questions}

【填空题】
{fill_in_questions}

【判断题】
{true_or_false_questions}

请按照以下维度进行全面的难度评估：

1. 单个题目难度评分
请对每道题目进行难度评分（1-10分），并说明理由：
   - 选择题难度评分
   - 填空题难度评分
   - 判断题难度评分

2. 整套题目的总体难度评分（1-10分）
   - 计算方法：所有题目难度分数的平均值
   - 给出详细的计算过程
   - 必须规范输出【最终分数】，例如4.32

3. 难度分布分析
   - 高难度题目（8-10分）
   - 中等难度题目（5-7分）
   - 低难度题目（1-4分）

4. 综合难度评估
   - 难度分布的合理性
   - 是否覆盖关键知识点
   - 与岗位要求的匹配程度

5. 改进建议
   - 题目难度平衡性
   - 知识点覆盖情况
   - 可能的调整方向
   
请在评估的末尾，给出最终评分，以下述格式给出：
**最终分数**：[分数]
"""

interview_DIFFICULT_PROMPT = """作为专业面试评估专家，请全面评估以下面试题目的难度。

岗位JD信息：
【任职要求】
{job_requirements}

【岗位职责】
{job_responsibilities}

面试题目（总计{questions_num}道）：

【面试题】
{interview_questions}

请按照以下维度进行全面的难度评估：

1. 单个题目难度评分
请对每道题目进行难度评分（1-10分），并说明理由：
   - 选择题难度评分
   - 填空题难度评分
   - 判断题难度评分

2. 整套题目的总体难度评分（1-10分）
   - 计算方法：所有题目难度分数的平均值
   - 给出详细的计算过程
   - 必须规范输出【最终分数】，例如4.32

3. 难度分布分析
   - 高难度题目（8-10分）
   - 中等难度题目（5-7分）
   - 低难度题目（1-4分）

4. 综合难度评估
   - 难度分布的合理性
   - 是否覆盖关键知识点
   - 与岗位要求的匹配程度

5. 改进建议
   - 题目难度平衡性
   - 知识点覆盖情况
   - 可能的调整方向

请在评估的末尾，给出最终评分，以下述格式给出：
**最终分数**：[分数]
"""

TEMPLATE_APP = "{\"score\":\"10\",\"reason\":\"满足全部要求\"}"

CORRECT_PROMPT = """
请对面试题目的正确性进行打分，打分时需按以下标准：1：满分是10分 2：确保题目中没有歧义、语法错误或逻辑漏洞，如果不满足，则扣掉3分，3：参考答案，需与题目要求完全匹配，如果不满足，则扣掉5分，4：题目表述应符合技术领域的标准术语，避免使用模糊或不规范的表达。如果不满足，则扣掉2分。打分结果需参考打分示例，其中score代表分数，reason代表扣分理由,扣分理由请使用中文，最终结果以json格式输出。<打分示例>{template_app}</打分示例>，<面试题目>{question}</面试题目>"
"""

RATIONALITY_PROMPT = """
请对面试题目的合理性进行打分，打分时需按以下标准：1：满分是10分 2：题目应能够反映真实的工作场景或实际项目需求，如果不满足，则扣掉4分，3：题目难度适中，一个中级面试者能在5分钟内完成，如果不满足，则扣掉4分。打分结果需参考打分示例，其中score_info代表分数，reason_info代表扣分理由,扣分理由请使用中文，最终结果以json格式输出。<打分示例>{template_app}</打分示例>，<面试题目>{question}</面试题目>
"""

PRACTICE_THEORY_PROMPT = (
    "你是一个题目评估助手。你的任务是根据提供的题目内容、答案，从**该题目是否能够有效考察候选人的问题解决能力和理论与实践结合能力**角度对题目进行评分。\n"
    "评分标准（1-10分），评估维度如下：\n"
    "1. **问题解决能力**：题目所涉及的问题是否考察出候选人解决实际问题的能力？\n"
    "2. **理论与实践结合**：题目是否要求候选人应用或解释相关的理论知识？\n"
    "评分结果需参考打分示例，其中score_info代表分数，reason_info代表扣分理由,扣分理由请使用中文，最终结果以json格式输出。<打分示例>{template_app}</打分示例>，<面试题目>{question}</面试题目>"
)

exam_DUPLICATION_PROMPT = '''作为一位专业的面试评估专家，请评估以下面试题的重复度。

面试题目：
1. 选择题：
{choice_questions}

2. 填空题：
{fill_in_questions}

3. 判断题：
{true_or_false_questions}

请按照以下固定格式评估题目重复度：

### 1. 整体重复度评分：**X分**
评分标准：
- 10分：题目完全不重复，知识点覆盖均衡
- 7-9分：极少量题目有相似点，但考察角度不同
- 4-6分：部分题目重复或相似度较高
- 1-3分：大量题目重复或考察相同知识点

**评分理由：**
[详细说明理由]

### 2. 具体重复分析：

**重复或相似题目分析：**
[列出具体重复或相似的题目]

**知识点覆盖分析：**
[分析知识点分布的均衡性]

### 总结：
- **重复度评分**：X分（补充说明）
- **主要问题**：[简要说明主要的重复问题]
- **改进建议**：[给出具体的改进建议]
'''

interview_DUPLICATION_PROMPT = '''作为一位专业的面试评估专家，请评估以下面试题的重复度。

面试题目：
{interview_questions}

请按照以下固定格式评估题目重复度：

### 1. 整体重复度评分：**X分**
评分标准：
- 10分：题目完全不重复，知识点覆盖均衡
- 7-9分：极少量题目有相似点，但考察角度不同
- 4-6分：部分题目重复或相似度较高
- 1-3分：大量题目重复或考察相同知识点

**评分理由：**
[详细说明理由]

### 2. 具体重复分析：

**重复或相似题目分析：**
[列出具体重复或相似的题目]

**知识点覆盖分析：**
[分析知识点分布的均衡性]

### 总结：
- **重复度评分**：X分（补充说明）
- **主要问题**：[简要说明主要的重复问题]
- **改进建议**：[给出具体的改进建议]
'''


def evaluate_jd_matching(jd_info, questions, judge_type):
    """Evaluation of the matching degree between the topic and JD"""
    try:
        # Preprocessing JD information
        # Use regular expressions to extract HTML content of job responsibilities and job requirements
        job_responsibilities_match = re.search(r'岗位职责：<p>(.*?)</p>', jd_info,
                                               re.DOTALL)
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
            job_requirements = re.sub(r'<[^>]+>', '\n', job_requirements_html)
        else:
            job_requirements = ""

        # Preprocessing question information
        if judge_type == "exam":
            choice_qs = format_questions(
                questions.get('choice_question') if questions else [])
            fill_qs = format_questions(
                questions.get('fill_in_question') if questions else [])
            tf_qs = format_questions(
                questions.get('true_or_false_question') if questions else [])
            prompt = exam_PIPEI_PROMPT.format(
                job_requirements=job_requirements,
                job_responsibilities=job_responsibilities,
                choice_questions=choice_qs,
                fill_in_questions=fill_qs,
                true_or_false_questions=tf_qs)
        else:
            prompt = interview_PIPEI_PROMPT.format(
                job_requirements=job_requirements,
                job_responsibilities=job_responsibilities,
                interview_questions=format_questions(
                    questions)) if questions else []

        max_retries = MAX_TRY  # Set the maximum number of retries
        attempt = 0
        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model=MODEL_TYPE,
                    messages=[{
                        "role": "system",
                        "content": "你是一个专业的面试评估专家，擅长评估面试题目与JD的匹配程度。"
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0)
                return response.choices[0].message.content

            except Exception as e:
                attempt += 1
                print(f"An error occurred during the evaluation: {str(e)}")
                print(f"Error Type: {type(e)}")
                print(f"Trying {attempt}th retry...")

                # If the maximum number of retries is reached, stop retrying
                if attempt >= max_retries:
                    print("The maximum number of retries has been reached, so retries are stopped.")
                    return ""  # Returns an empty string to indicate that the build failed.

                # Retry after 1 second delay
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred during the overall assessment: {str(e)}")
        print(f"Error Type: {type(e)}")
        return ""  # If the overall evaluation fails, an empty string is also returned

def evaluate_question_difficulty(jd_info, questions, judge_type):
    try:
        # Preprocessing JD information
        # Use regular expressions to extract HTML content of job responsibilities and job requirements
        job_responsibilities_match = re.search(r'岗位职责：<p>(.*?)</p>', jd_info,
                                               re.DOTALL)
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
            job_requirements = re.sub(r'<[^>]+>', '\n', job_requirements_html)
        else:
            job_requirements = ""

        # Preprocessing question information
        if judge_type == "exam":
            choice_qs = format_questions(
                questions.get('choice_question') if questions else [])
            fill_qs = format_questions(
                questions.get('fill_in_question') if questions else [])
            tf_qs = format_questions(
                questions.get('true_or_false_question') if questions else [])
            prompt = exam_DIFFICULT_PROMPT.format(
                job_requirements=job_requirements,
                job_responsibilities=job_responsibilities,
                questions_num=len(choice_qs) + len(fill_qs) + len(tf_qs),
                choice_questions=choice_qs,
                fill_in_questions=fill_qs,
                true_or_false_questions=tf_qs)
        else:
            prompt = interview_DIFFICULT_PROMPT.format(
                job_requirements=job_requirements,
                job_responsibilities=job_responsibilities,
                questions_num=len(questions) if questions else 0,
                interview_questions=format_questions(questions)
                if questions else [])

        # Call DeepSeek API and add fault-tolerant retry mechanism
        max_retries = MAX_TRY  # Set the maximum number of retries
        attempt = 0
        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model=MODEL_TYPE,
                    messages=[{
                        "role": "system",
                        "content": "你是一个专业的面试评估专家，擅长评估面试题目的整体难度。"
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0)
                return response.choices[0].message.content

            except Exception as e:
                attempt += 1
                print(f"An error occurred during the evaluation: {str(e)}")
                print(f"Error Type: {type(e)}")
                print(f"Trying {attempt}th retry...")

                # If the maximum number of retries is reached, stop retrying
                if attempt >= max_retries:
                    print("The maximum number of retries has been reached, so retries are stopped.")
                    return None

                # Retry after 1 second delay
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred during the overall evaluation: {str(e)}")
        print(f"Error Type:{type(e)}")
        return None


def evaluate_question_duplication(questions, judge_type):
    try:
        prompt = ""
        # Preprocessing question information
        if judge_type == "exam":
            choice_qs = format_questions(
                questions.get('choice_question') if questions else [])
            fill_qs = format_questions(
                questions.get('fill_in_question') if questions else [])
            tf_qs = format_questions(
                questions.get('true_or_false_question') if questions else [])
            prompt = exam_DUPLICATION_PROMPT.format(
                choice_questions=choice_qs,
                fill_in_questions=fill_qs,
                true_or_false_questions=tf_qs)
        else:
            prompt = interview_DUPLICATION_PROMPT.format(
                interview_questions=format_questions(questions
                                                   ) if questions else [])

        # Call DeepSeek API and add fault-tolerant retry mechanism
        max_retries = MAX_TRY  # Set the maximum number of retries
        attempt = 0
        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model=MODEL_TYPE,
                    messages=[{
                        "role": "system",
                        "content": "你是一位专业的面试评估专家，擅长评估面试题目的重复度和知识点覆盖情况。"
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0)
                return response.choices[0].message.content

            except Exception as e:
                attempt += 1
                print(f"An error occurred during the evaluation: {str(e)}")
                print(f"Error Type: {type(e)}")
                print(f"Trying {attempt}th retry...")

                # If the maximum number of retries is reached, stop retrying
                if attempt >= max_retries:
                    print("The maximum number of retries has been reached, so retries are stopped.")
                    return None

                # Retry after 1 second delay
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred during the overall assessment: {str(e)}")
        print(f"Error Type: {type(e)}")
        return None


def evaluate_question_practice_theory(questions, judge_type):
    try:
        prompt = ""
        # Preprocessing question information
        if judge_type == "exam":
            choice_qs = format_questions(
                questions.get('choice_question') if questions else [])
            fill_qs = format_questions(
                questions.get('fill_in_question') if questions else [])
            tf_qs = format_questions(
                questions.get('true_or_false_question') if questions else [])
            prompt = exam_DUPLICATION_PROMPT.format(
                choice_questions=choice_qs,
                fill_in_questions=fill_qs,
                true_or_false_questions=tf_qs)
        else:
            prompt = interview_DUPLICATION_PROMPT.format(
                interview_questions=format_questions(questions
                                                   ) if questions else [])

        # Call DeepSeek API and add fault-tolerant retry mechanism
        max_retries = MAX_TRY  # Set the maximum number of retries
        attempt = 0
        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model=MODEL_TYPE,
                    messages=[{
                        "role": "system",
                        "content": "你是一位专业的面试评估专家，擅长评估面试题目的理论和实际结合程度。"
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0)
                return response.choices[0].message.content

            except Exception as e:
                attempt += 1
                print(f"An error occurred during the evaluation: {str(e)}")
                print(f"Error Type: {type(e)}")
                print(f"Trying {attempt}th retry...")

                # If the maximum number of retries is reached, stop retrying
                if attempt >= max_retries:
                    print("The maximum number of retries has been reached, so retries are stopped.")
                    return None

                # Retry after 1 second delay
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred during the overall evaluation:{str(e)}")
        print(f"Error Type: {type(e)}")
        return None


def format_questions(questions):
    # If the question does not exist, return null.
    if questions is None:
        return ""
    """Formatting the topic list"""
    formatted = []
    for i, q in enumerate(questions, 1):
        question_text = f"Q{i}: {q['question']}"
        if 'options' in q:
            options = [
                f"{chr(65+j)}. {opt}" for j, opt in enumerate(q['options'])
            ]
            question_text += "\n" + "\n".join(options)
        if 'answer' in q:
            question_text += f"\n答案: {q['answer']}"
        formatted.append(question_text)
    return "\n\n".join(formatted)


def evaluate_questions(jd_infos,
                       question_list,
                       judge_type="exam",
                       metric="match"):
    """Evaluates the stored questions and returns a list of results"""
    evaluation_results = []

    if metric == "match":
        def process_single(args):
            idx, jd, questions, judge_type = args
            evaluation_result = evaluate_jd_matching(jd, questions, judge_type)
            return idx, evaluation_result if evaluation_result else None

        task_args = [
            (idx, jd, questions, judge_type)
            for idx, (jd, questions) in enumerate(zip(jd_infos, question_list))
        ]
        evaluation_results = [None] * len(task_args)  # Preallocate result list
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            # Submit all tasks (non-blocking)
            futures = [
                executor.submit(process_single, args) for args in task_args
            ]

            # Create a progress bar (updates in real time)
            with tqdm(total=len(futures), desc="Evaluating") as pbar:
                # Process results in order of completion (but maintain original order)
                for future in as_completed(futures):
                    idx, result = future.result()
                    evaluation_results[idx] = result  # Write to the correct position by index
                    pbar.update(1)  # Update the progress bar each time a task is completed
    elif metric == "difficulty":
        def process_single(args):
            idx, jd, questions, judge_type = args
            evaluation_result = evaluate_question_difficulty(
                jd, questions, judge_type)
            return idx, evaluation_result if evaluation_result else None

        task_args = [
            (idx, jd, questions, judge_type)
            for idx, (jd, questions) in enumerate(zip(jd_infos, question_list))
        ]
        evaluation_results = [None] * len(task_args)  # Preallocate result list
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            # Submit all tasks (non-blocking)
            futures = [
                executor.submit(process_single, args) for args in task_args
            ]

            # Create a progress bar (updates in real time)
            with tqdm(total=len(futures), desc="Evaluating") as pbar:
                # Process results in order of completion (but maintain original order)
                for future in as_completed(futures):
                    idx, result = future.result()
                    evaluation_results[idx] = result  # Write to the correct position by index
                    pbar.update(1)  # Update the progress bar each time a task is completed
    elif metric == "duplication":
        def process_single(args):
            idx, jd, questions, judge_type = args
            evaluation_result = evaluate_question_duplication(
                questions, judge_type)
            return idx, evaluation_result if evaluation_result else None

        task_args = [
            (idx, jd, questions, judge_type)
            for idx, (jd, questions) in enumerate(zip(jd_infos, question_list))
        ]
        evaluation_results = [None] * len(task_args)  # Preallocate result list
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            # Submit all tasks (non-blocking)
            futures = [
                executor.submit(process_single, args) for args in task_args
            ]

            # Create a progress bar (updates in real time)
            with tqdm(total=len(futures), desc="Evaluating") as pbar:
                # Process results in order of completion (but maintain original order)
                for future in as_completed(futures):
                    idx, result = future.result()
                    evaluation_results[idx] = result  # Write to the correct position by index
                    pbar.update(1)  # Update the progress bar each time a task is completed
    elif metric == "practice_theory":
        for questions in tqdm(question_list):
            evaluation_result = evaluate_question_practice_theory(
                questions, judge_type)
            if evaluation_result:
                evaluation_results.append(evaluation_result)
            else:
                evaluation_results.append(None)
    else:
        pass

    return evaluation_results


def parse_match_score(evaluation_text):
    if evaluation_text is None:
        return 0
    """Parse the evaluation result text and extract key information"""
    try:
        # Extracting the overall match score
        final_score_match = re.search(r'\*\*最终分数\*\*[：:]\s*(\d+\.?\d*)分?',
                                      evaluation_text, re.DOTALL)
        if final_score_match:
            match_score = float(final_score_match.group(1))

        if not final_score_match:
            # Try a different format
            final_score_match = re.search(r'最终分数[\""\*]*：\s*(\d+\.?\d*)',
                                          evaluation_text, re.DOTALL)
            if final_score_match:
                match_score = float(final_score_match.group(1))
            else:
                match_score = 0

        # print(f"The parsed matching score: {match_score}")  # Add debug output
        #print(f"Original text snippet: {evaluation_text[:200]}...")  # Add debug output

        return match_score

    except Exception as e:
        print(f"Error parsing evaluation results: {str(e)}")
        print(f"Error Type: {type(e)}")
        #print(f"Original text: {evaluation_text[:200]}...")
        return 0


def extrac_questions_list(recruitai_data, baseline_data,
                          judge_type) -> list[list]:
    # Put each question of each question type into the list
    # Each question is scored individually and then the average score is calculated at the end
    recruitai_question_lists = []
    baseline_question_lists = []
    if judge_type == "exam":
        for h_data, b_data in zip(recruitai_data, baseline_data):
            question_lists = []
            if h_data is None:
                recruitai_question_lists.append([])
                continue
            else:
                for questions in h_data.values():
                    if questions is None:
                        continue
                    for item in questions:
                        if "options" in item:
                            options = [
                                f"{chr(65+j)}. {opt}"
                                for j, opt in enumerate(item['options'])
                            ]
                            options = "\n".join(options)
                            question = f'题目: {item["question"] if "question" in item else None}\n{options}\n答案: {item["answer"] if "answer" in item else None}'
                        else:
                            question = f'题目: {item["question"] if "question" in item else None}\n答案: {item["answer"] if "answer" in item else None}'
                        question_lists.append(question)
            recruitai_question_lists.append(question_lists)
            question_lists = []
            if b_data is None:
                baseline_question_lists.append([])
                continue
            else:
                for questions in b_data.values():
                    if questions is None:
                        continue
                    for item in questions:
                        if "options" in item:
                            options = [
                                f"{chr(65+j)}. {opt}"
                                for j, opt in enumerate(item['options'])
                            ]
                            options = "\n".join(options)
                            question = question = f'题目: {item["question"] if "question" in item else None}\n{options}\n答案: {item["answer"] if "answer" in item else None}'
                        else:
                            question = question = f'题目: {item["question"] if "question" in item else None}\n答案: {item["answer"] if "answer" in item else None}'
                        question_lists.append(question)
            baseline_question_lists.append(question_lists)
    else:  #
        for data in recruitai_data:
            question_lists = []
            if data is None:
                recruitai_question_lists.append(question_lists)
                continue
            for item in data:
                question = f'题目: {item["question"] if "question" in item else None}\n答案: {item["answer"] if "answer" in item else None}'
                question_lists.append(question)
            recruitai_question_lists.append(question_lists)
        for data in baseline_data:
            question_lists = []
            if data is None:
                baseline_question_lists.append(question_lists)
                continue
            for item in data:
                question = f'题目: {item["question"] if "question" in item else None}\n答案: {item["answer"] if "answer" in item else None}'
                question_lists.append(question)
            baseline_question_lists.append(question_lists)
    return recruitai_question_lists, baseline_question_lists


def parse_difficulty_score(evaluation_text):
    if evaluation_text is None:
        return -1

    try:
        # Check if the text contains "Final Score"
        if "最终分数" in evaluation_text:
            # Define regular matching pattern
            difficulty_patterns = [
                r'\*\*最终分数\*\*[：:]\s*(\d+\.?\d*)分?',  # **Final score**: 4.27
                r'\*\*最终分数\*\*[：:]\s*\*\*(\d+\.?\d*)\*\*'  # **Final score**: 4.27
            ]

            difficulty_score = None

            # Try to use regular pattern to match the final score
            for pattern in difficulty_patterns:
                score_match = re.search(pattern, evaluation_text, re.DOTALL)
                if score_match:
                    difficulty_score = float(score_match.group(1))
                    break  # If a match is found, break out of the loop

            # If there is no match for **final score**, extract the last number in the text to the right of "final score"
            if difficulty_score is None:
                # Extract all text to the right of "Final Score"
                difficulty_text = evaluation_text.split("最终分数")[-1]

                # Find numbers from extracted text sections
                all_numbers = re.findall(r'\d+\.?\d*',
                                         difficulty_text)  # Extract all numbers
                if all_numbers:
                    difficulty_score = float(all_numbers[-1])  # Take the last number
                else:
                    difficulty_score = -1  # If there is no number, returns -1

            return difficulty_score
        else:
            return -1  # If there is no "final score" in the text, return -1

    except Exception as e:
        print(f"Error parsing evaluation results: {str(e)}")
        print(f"Error Type: {type(e)}")
        return -1


def parse_duplication_result(evaluation_text):
    if not evaluation_text:
        return -1
    """Analyzing the repeatability assessment results"""
    try:
        # Extract the duplication score (actually the uniqueness score)
        duplication_score = None
        score_match = re.search(r'整体重复度评分：\*\*(\d+)分\*\*', evaluation_text,
                                re.DOTALL)
        if score_match:
            duplication_score = int(score_match.group(1))

        # Extract rating reasons
        score_reason = ""
        reason_match = re.search(r'\*\*评分理由：\*\*\n(.*?)\n\n', evaluation_text,
                                 re.DOTALL)
        if reason_match:
            score_reason = reason_match.group(1).strip()

        # Extract replicate analysis
        duplication_analysis = ""
        analysis_match = re.search(r'\*\*重复或相似题目分析：\*\*\n(.*?)\n\n',
                                   evaluation_text, re.DOTALL)
        if analysis_match:
            duplication_analysis = analysis_match.group(1).strip()

        # Modify the statement of the rating reason
        final_reasoning = f"独特度: {duplication_score}分 (分数越高表示题目越独特) | 评分理由: {score_reason} | 重复分析: {duplication_analysis}"

        return duplication_score if duplication_score else -1

    except Exception as e:
        print(f"Error parsing evaluation results: {str(e)}")
        return -1


def get_ideal_difficulty_range(job_level):
    if job_level == '高级':
        return (8.0, 10.0)
    elif job_level == '中级':
        return (5.0, 7.9)
    else:
        return (1.0, 4.9)


def calculate_match_score(job_level, difficulty_score):
    if difficulty_score == -1:
        return -1
    # Returns 1 for a good match, -1 for a bad match
    ideal_range = get_ideal_difficulty_range(job_level)

    # Calculate the relative distance of the difficulty score to the ideal range
    distance_from_ideal = max(0, difficulty_score - ideal_range[1],
                              ideal_range[0] - difficulty_score)

    # According to the distance, a piecewise linear function is used to give the fit score.
    if distance_from_ideal == 0:
        # return 10
        return 1
    elif distance_from_ideal <= 1:
        # return 8
        return -1
    elif distance_from_ideal <= 2:
        # return 6
        return -1
    elif distance_from_ideal <= 3:
        # return 4
        return -1
    else:
        # return 2
        return -1


def get_llm_response(input_text):
    messages = [{
        "role": "system",
        "content": "你是一个面试考官，请逐步思考并简洁地回答我的问题",
    }, {
        "role": "user",
        "content": input_text,
    }]
    final_content = ''
    use_stream = False
    max_retries = MAX_TRY  # Set the maximum number of retries
    attempt = 0

    while attempt < max_retries:
        try:
            # Making a request
            response = client.chat.completions.create(model=MODEL_TYPE,
                                                      messages=messages,
                                                      stream=use_stream,
                                                      temperature=0,
                                                      presence_penalty=1.1,
                                                      frequency_penalty=1.1,
                                                      top_p=0.9)

            # Processing response content
            if response:
                if use_stream:
                    for chunk in response:
                        print(chunk.choices[0].delta.content)
                else:
                    content = response.choices[0].message.content
                    final_content = content
                    return final_content
            else:
                return final_content

        except Exception as e:
            attempt += 1
            print(f"Request error: {str(e)}")
            print(f"Error Type: {type(e)}")
            print(f"Trying {attempt}th retry...")

            # If the maximum number of retries is reached, stop retrying
            if attempt >= max_retries:
                print("The maximum number of retries has been reached, so retries are stopped.")
                return final_content  # Returning empty content indicates failure

            # Retry after 1 second delay
            time.sleep(1)


def clean_response(final_content):
    start_pos = final_content.find('```json')
    if start_pos >= 0:
        tail_content = final_content[start_pos + 7:]
        ##print("tail_content======",tail_content)
        end_pos = tail_content.find('```')
        if (end_pos > 0):
            ss_content = tail_content[:end_pos]
            return ss_content
        else:
            return final_content
    else:
        return final_content


from concurrent.futures import ThreadPoolExecutor, as_completed


def calculate_score_generator(prompt_template, progress_desc):

    def process_item(item):
        try:
            output = get_llm_response(
                prompt_template.format(template_app=TEMPLATE_APP,
                                       question=item))
            output = clean_response(output)
            try:
                cur = json.loads(output.strip())
                return {
                    'status': 'success',
                    'score': int(cur['score']),
                    'output': output
                }
            except Exception as e:
                print("=========Parsing Error=========")
                print(output.strip())
                return {'status': 'parse_error', 'output': ""}
        except Exception as e:
            print(e)
            return {'status': 'error', 'output': ""}

    def inner_func(datas):
        score_list = []
        outputs = []
        with ThreadPoolExecutor() as executor:
            for data in tqdm(datas, desc=progress_desc):
                futures = [
                    executor.submit(process_item, item) for item in data
                ]
                scores = []
                out = []

                # The key to keeping order: get results in the order they were submitted
                for future in futures:
                    result = future.result()
                    if result['status'] == 'success':
                        scores.append(result['score'])
                        out.append(result['output'])
                    else:
                        scores.append(-1)
                        out.append(result['output'])

                score_list.append(scores)
                outputs.append(out)
        return score_list, outputs

    return inner_func


calculate_correctness = calculate_score_generator(CORRECT_PROMPT,
                                                  "evaluate correct...")
calculate_rationality = calculate_score_generator(RATIONALITY_PROMPT,
                                                  "evaluate rationality...")


def calculate_practice_theory(datas):
    score_list = []
    outputs = []

    def process_item(args):
        """Thread function that processes a single item"""
        j, item = args  # Receive item and its index j in data
        try:
            output = get_llm_response(
                PRACTICE_THEORY_PROMPT.format(template_app=TEMPLATE_APP,
                                              question=item))
            output = clean_response(output)
            try:
                cur = json.loads(output.strip())
                score = int(cur['score'])
                return (j, score, output)
            except Exception as e:
                print("=========Parsing Error=========")
                print(output.strip())
                return (j, -1, output)
        except Exception as e:
            print(e)
            return (j, -1, "")

    # The outer loop keeps the original order of datas
    for data in tqdm(datas, desc="evaluate practice theory..."):
        n_items = len(data)
        # Pre-initialize the result container of the current data
        scores = [-1] * n_items
        out = [""] * n_items

        # Create a thread pool for the current data
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=128) as executor:
            # Submit all item tasks of the current data (with original index j)
            futures = [
                executor.submit(process_item, (j, item))
                for j, item in enumerate(data)
            ]

            # Collect results in order of completion (by j locating the storage location)
            for future in concurrent.futures.as_completed(futures):
                try:
                    j, score, output = future.result()
                    scores[j] = score
                    out[j] = output
                except Exception as e:
                    print(f"Handling failure: {str(e)}")

        score_list.append(scores)
        outputs.append(out)

    return score_list, outputs


def avg_list(data):
    # Filter out 0 values
    filtered_data = [x for x in data if x != -1]
    if len(filtered_data) == 0:
        return 0
    return round(sum(filtered_data) / len(filtered_data), 2)


def cal_win_rate(data1, data2):
    win_rates = []
    for score1, score2 in zip(data1, data2):
        if score1 > score2:
            # recruitai wins and gets one point
            win_rates.append(1)
        elif score1 == score2:  #Average 0 points
            win_rates.append(0)
        else:
            win_rates.append(-1)
    return win_rates


def match_evaluate(args, jd_infos, recruitai_data, baseline_data, eval_data):
    # Compatibility assessment Use JD + Question for assessment
    print("Conduct a compatibility assessment!")
    match_recruitai_results = evaluate_questions(jd_infos, recruitai_data,
                                                args.judge_type, "match")
    match_baseline_results = evaluate_questions(jd_infos, baseline_data,
                                                args.judge_type, "match")

    print("Analytical matching evaluation")
    recruitai_score = [
        parse_match_score(item) for item in match_recruitai_results
    ]
    baseline_score = [
        parse_match_score(item) for item in match_baseline_results
    ]
    # Save the results
    match_win_results = cal_win_rate(recruitai_score, baseline_score)
    for idx, data in enumerate(eval_data):
        data["match_eval_results"] = {
            "recruitai": match_recruitai_results[idx],
            "baseline": match_baseline_results[idx]
        }
        data["match_score"] = {
            "recruitai": recruitai_score[idx],
            "baseline": baseline_score[idx]
        }
        data["match_win_rates"] = match_win_results[idx]
    return eval_data


def difficulty_level_evaluate(args, jd_infos, recruitai_data, baseline_data,
                              eval_data, difficulty_level):
    # Question difficulty assessment - with fit score
    print("Conduct a difficulty assessment")
    difficulty_recruitai_results = evaluate_questions(jd_infos, recruitai_data,
                                                     args.judge_type,
                                                     "difficulty")
    difficulty_baseline_results = evaluate_questions(jd_infos, baseline_data,
                                                     args.judge_type,
                                                     "difficulty")
    print("Parsing Difficulty Rating")
    recruitai_difficulty_score = [
        parse_difficulty_score(item) for item in difficulty_recruitai_results
    ]
    baseline_difficulty_score = [
        parse_difficulty_score(item) for item in difficulty_baseline_results
    ]

    recruitai_score = [
        calculate_match_score(difficulty_level[idx],
                              recruitai_difficulty_score[idx])
        for idx in range(len(difficulty_level))
    ]
    baseline_score = [
        calculate_match_score(difficulty_level[idx],
                              baseline_difficulty_score[idx])
        for idx in range(len(difficulty_level))
    ]
    # Save the results
    difficulty_win_results = cal_win_rate(recruitai_score, baseline_score)
    for idx, data in enumerate(eval_data):
        data["difficulty_eval_results"] = {
            "recruitai": difficulty_recruitai_results[idx],
            "baseline": difficulty_baseline_results[idx]
        }
        data["difficulty_score"] = {
            "recruitai": recruitai_score[idx],
            "baseline": baseline_score[idx]
        }
        data["difficulty_win_rates"] = difficulty_win_results[idx]
    return eval_data


def correctness_rationality_evaluate(args, recruitai_data, baseline_data,
                                     eval_data):
    # Correctness and rationality of technical interview questions
    # First extract the title of each question, and then apply prompt to calculate the correctness score and rationality score respectively
    recruitai_question_lists, baseline_question_lists = extrac_questions_list(
        recruitai_data, baseline_data, args.judge_type)
    # Use prompt to calculate correctness score and rationality score
    # First calculate the correctness score:
    hagonda_correct_score_list, recruitai_correctness_outputs = calculate_correctness(
        recruitai_question_lists)
    baseline_correct_score_list, baseline_correctness_outputs = calculate_correctness(
        baseline_question_lists)
    # Recalculate the plausibility score
    recruitai_rationality_score_list, recruitai_rationality_outputs = calculate_rationality(
        recruitai_question_lists)
    baseline_rationality_score_list, baseline_rationality_outputs = calculate_rationality(
        baseline_question_lists)

    recruitai_score = [
        round(0.7 * avg_list(c) + 0.3 * avg_list(r), 2) for c, r in zip(
            hagonda_correct_score_list, recruitai_rationality_score_list)
    ]
    baseline_score = [
        round(0.7 * avg_list(c) + 0.3 * avg_list(r), 2) for c, r in zip(
            baseline_correct_score_list, baseline_rationality_score_list)
    ]

    c_r_win_rates = cal_win_rate(recruitai_score, baseline_score)

    for idx, data in enumerate(eval_data):
        data["correct_score_list"] = {
            "recruitai": hagonda_correct_score_list[idx],
            "baseline": baseline_correct_score_list[idx]
        }
        data["correct_model_outputs"] = {
            "recruitai": recruitai_correctness_outputs[idx],
            "baseline": baseline_correctness_outputs[idx]
        }
        data["rationality_model_outputs"] = {
            "recruitai": recruitai_rationality_outputs[idx],
            "baseline": baseline_rationality_outputs[idx]
        }
        data["rationality_score_list"] = {
            "recruitai": recruitai_rationality_score_list[idx],
            "baseline": baseline_rationality_score_list[idx]
        }
        data["correct_rationality_total_score"] = {
            "recruitai": recruitai_score[idx],
            "baseline": baseline_score[idx]
        }
        data["c_r_win_rate"] = c_r_win_rates[idx]

    return eval_data


def practice_theory_evaluate(args, recruitai_data, baseline_data, eval_data):

    recruitai_question_lists, baseline_question_lists = extrac_questions_list(
        recruitai_data, baseline_data, args.judge_type)

    hagonda_practice_theory_score_list, recruitai_practice_theory_outputs = calculate_practice_theory(
        recruitai_question_lists)
    baseline_practice_theory_score_list, baseline_practice_theory_outputs = calculate_practice_theory(
        baseline_question_lists)

    recruitai_score = [
        round(avg_list(c), 2) for c in hagonda_practice_theory_score_list
    ]
    baseline_score = [
        round(avg_list(c), 2) for c in baseline_practice_theory_score_list
    ]

    practice_theory_win_rates = cal_win_rate(recruitai_score, baseline_score)

    for idx, data in enumerate(eval_data):
        data["practice_theory_score_list"] = {
            "recruitai": hagonda_practice_theory_score_list[idx],
            "baseline": baseline_practice_theory_score_list[idx]
        }
        data["practice_theory_model_outputs"] = {
            "recruitai": recruitai_practice_theory_outputs[idx],
            "baseline": baseline_practice_theory_outputs[idx]
        }
        data["practice_theory_total_score"] = {
            "recruitai": recruitai_score[idx],
            "baseline": baseline_score[idx]
        }
        data["practice_theory_win_rates"] = practice_theory_win_rates[idx]
    return eval_data


def duplication_evalutate(args, jd_infos, recruitai_data, baseline_data,
                          eval_data):
    # Question repetition assessment
    print("Conduct repeatability assessment")
    duplication_recruitai_results = evaluate_questions(jd_infos, recruitai_data,
                                                      args.judge_type,
                                                      "duplication")
    duplication_baseline_results = evaluate_questions(jd_infos, baseline_data,
                                                      args.judge_type,
                                                      "duplication")
    print("Parsing repetition scores")
    recruitai_duplication_score = [
        parse_duplication_result(item) for item in duplication_recruitai_results
    ]
    baseline_duplication_score = [
        parse_duplication_result(item) for item in duplication_baseline_results
    ]

    # Save the results
    duplication_win_results = cal_win_rate(recruitai_duplication_score,
                                           baseline_duplication_score)
    for idx, data in enumerate(eval_data):
        data["duplication_eval_results"] = {
            "recruitai": duplication_recruitai_results[idx],
            "baseline": duplication_baseline_results[idx]
        }
        data["duplication_score"] = {
            "recruitai": recruitai_duplication_score[idx],
            "baseline": baseline_duplication_score[idx]
        }
        data["duplication_win_rates"] = duplication_win_results[idx]
    return eval_data


def main(args):
    # File Reading
    print("File Reading")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    if args.our_for_match_file is not None:
        with open(args.our_for_match_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        if args.debug:
            datas = datas[:3]
            eval_data = eval_data[:3]
            args.output_file = "./debug.json"
    print("Pre-extraction")
    jd_infos = [item['jd_info'] for item in eval_data]
    difficulty_level = [item['post_level'] for item in eval_data]
    recruitai_data = [item['our'] for item in eval_data]
    baseline_data = [item['baseline'] for item in eval_data]

    recruitai_data_for_match = recruitai_data
    if args.our_for_match_file is not None:
        recruitai_data_for_match = [item['baseline'] for item in datas]

    for metric in args.metrics:
        if metric == "match":
            eval_data = match_evaluate(args, jd_infos, recruitai_data_for_match,
                                       baseline_data, eval_data)

        elif metric == "difficulty_level":
            eval_data = difficulty_level_evaluate(args, jd_infos,
                                                  recruitai_data, baseline_data,
                                                  eval_data, difficulty_level)
        elif metric == "correctness_rationality":
            eval_data = correctness_rationality_evaluate(
                args, recruitai_data, baseline_data, eval_data)

        elif metric == "duplication":
            eval_data = duplication_evalutate(args, jd_infos, recruitai_data,
                                              baseline_data, eval_data)

        elif metric == "practice_theory":
            eval_data = practice_theory_evaluate(args, recruitai_data,
                                                 baseline_data, eval_data)

        else:
            raise RuntimeError("Wrong metric used")

    # Save the results：
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        default="both_interview.json",
                        help="The file to be evaluated contains the results of baseline and ours")
    parser.add_argument("--our_for_match_file", default=None, type=str)
    parser.add_argument("--output_file",
                        type=str,
                        default="evaled_both_interview.json",
                        help="Evaluation result saving path")
    parser.add_argument("--judge_type",
                        type=str,
                        default="interview",
                        help="Distinguish whether the questions to be assessed are written test questions or interview questions")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--metrics",
        nargs='+',
        type=str,
        help=
        "choice option: match/difficulty_level/correctness_rationality/duplication"
    )
    args = parser.parse_args()
    main(args)

import json

from openai import OpenAI, APIConnectionError, APIError

from balance_check import get_balance
from get_key import get_deepseek_key

client = OpenAI(api_key=get_deepseek_key(), base_url="")

file_path = "random_jd.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

balance1 = get_balance()
print("Current CNY balance:", balance1)
answer = []
total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
for i in range(8):
    labels = data[i * 25 + 24]["label"]
    labels = labels.replace("无,", "").replace("无", "").strip().strip(",")
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system",
                 "content": """你是一名专业的HR，我将提供一组从JD（职位描述）中提取出的技术标签。
                                            你的任务是基于以下三个维度对标签集进行优化：
                                            
                                            1. **正确性**：移除所有与技术无关的软技能标签（如性格、沟通能力等），仅保留适用于技术面试的硬技能标签。
                                            2. **重复性**：检查并去除重复标签，确保每个标签唯一,对于相似度高的标签仅保留其一。
                                            3. **多样性**：评估标签集是否涵盖核心技术领域，并在必要时补充遗漏的关键技术标签，以提高技术覆盖面。
                                            
                                            ### **输出要求**
                                            
                                            - **仅输出优化后的技术标签**，标签之间用英文逗号 `,` 分隔。
                                            - **严格按照要求输出，不得包含额外内容**。
                                            
                                            #### **示例**
                                            
                                            **输入**：
                                            
                                            标签集：Python, python, 沟通能力, Java, Java, 团队协作, SQL
                                            
                                            **输出**：
                                            
                                            Python,Java,SQL"""},
                {"role": "user", "content": f"标签:{labels}"},
            ],
            stream=False
        )
        print(response.choices[0].message.reasoning_content)
        print(response.choices[0].message.content)
        answer.append(response.choices[0].message.content)
        usage = response.usage
        if usage:
            total_usage["prompt_tokens"] += usage.prompt_tokens
            total_usage["completion_tokens"] += usage.completion_tokens
            total_usage["total_tokens"] += usage.total_tokens

    except APIConnectionError as e:
        print("Connection Error:", e.__cause__)
    except APIError as e:
        print("API return error:", e)

try:
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system",
             "content": """你是一名专业的HR，我将提供一组从JD（职位描述）中提取出的技术标签。
                                        你的任务是基于以下三个维度对标签集进行优化：

                                        1. **正确性**：移除所有与技术无关的软技能标签（如性格、沟通能力等），仅保留适用于技术面试的硬技能标签。
                                        2. **重复性**：检查并去除重复标签，确保每个标签唯一,对于相似度高的标签仅保留其一。
                                        3. **多样性**：评估标签集是否涵盖核心技术领域，并在必要时补充遗漏的关键技术标签，以提高技术覆盖面。

                                        ### **输出要求**

                                        - **仅输出优化后的技术标签**，标签之间用英文逗号 `,` 分隔。
                                        - **严格按照要求输出，不得包含额外内容**。

                                        #### **示例**

                                        **输入**：

                                        标签集：Python, python, 沟通能力, Java, Java, 团队协作, SQL

                                        **输出**：

                                        Python,Java,SQL"""},
            {"role": "user", "content": f"标签:{','.join(answer)}"},
        ],
        stream=False
    )
    print(response.choices[0].message.reasoning_content)
    print(response.choices[0].message.content)
    labels = response.choices[0].message.content.split(',')
    with open("jd_final_label.json", 'w', encoding='utf-8') as file:
        json.dump(labels, file, ensure_ascii=False, indent=4)
    usage = response.usage
    if usage:
        total_usage["prompt_tokens"] += usage.prompt_tokens
        total_usage["completion_tokens"] += usage.completion_tokens
        total_usage["total_tokens"] += usage.total_tokens

except APIConnectionError as e:
    print("Connection Error:", e.__cause__)
except APIError as e:
    print("API return error:", e)

print(total_usage)

balance2 = get_balance()
print("Current CNY balance:", balance2, "Expected Use:", float(balance1) - float(balance2))

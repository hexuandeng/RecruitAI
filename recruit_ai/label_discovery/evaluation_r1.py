import json

from openai import OpenAI, APIConnectionError, APIError

import get_key

client = OpenAI(api_key=get_key.get_deepseek_key(), base_url="")


with open("jd_label3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

label = ','.join(data)
print(label)

total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
try:
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system",
             "content": """你是一名专业的 HR，我将提供一组从该 JD 中提取出的技术标签。  

                                    你的任务是基于以下三个维度对标签进行评分：  
                                    
                                    1. **正确性（满分 100 分）**：评估这些标签是否适合作为技术面试中的技术考察点。  
                                       - 每出现一个与技术无关的软技能（如性格、沟通能力）**扣 10 分**。  
                                       - 技术相关但过于模糊或不常见（如 "计算机知识"）**扣 5 分**。  
                                       - 所有标签都与技术相关，得 100 分。  
                                       - 过多标签（20个以上）和技术无关，不超过50分。
                                    
                                    2. **重复性（满分 100 分）**：检查标签中是否存在重复项。  
                                       - 完全无重复，得 100 分。  
                                       - **每个重复标签扣 1 分**。  
                                       - 过多标签重复，最高不得超过 50 分。  
                                    
                                    3. **多样性（满分 100 分）**：衡量标签是否涵盖了足够广泛的技术领域。  
                                    ---
                                    
                                    ### **输出要求**  
                                    
                                    - **第一行**：解释你的评分依据，包含对正确性、重复性和多样性的分析。  
                                    - **第二行**：仅输出三个分数，以逗号 `,` 隔开，格式如下：  
                                    
                                      `87,95,91`  
                                    
                                    - **示例**：
                                      - **输入**：`["Python", "Java", "沟通能力", "Python", "深度学习"]`  
                                      - **输出**：
                                        正确性得分 90（沟通能力与技术无关扣 10 分），重复性得分 99（Python 重复一次扣 1 分），多样性得分 80（涵盖编程语言和 AI 领域）。
                                        90,99,80
                                    
                                    - **严格按照要求输出**，不得输出额外内容。 """},
            {"role": "user", "content": f"标签:{label}"},
        ],
        stream=False
    )
    print(response.choices[0].message.content)
    usage = response.usage
    if usage:
        total_usage["prompt_tokens"] += usage.prompt_tokens
        total_usage["completion_tokens"] += usage.completion_tokens
        total_usage["total_tokens"] += usage.total_tokens
    print(total_usage)

except APIConnectionError as e:
    print("Connection Error:", e.__cause__)
except APIError as e:
    print("API returns error:", e)

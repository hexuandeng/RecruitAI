import json
import time
from balance_check import get_balance
from openai import OpenAI, APIConnectionError, APIError

from get_key import get_deepseek_key

client = OpenAI(api_key=get_deepseek_key(), base_url="")

balance1 = get_balance()
print("Current CNY balance:", balance1)

# Record total API usage
total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

file_path = 'random_jd.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

print("Unused data is being processed...")
print(f"The total number of data to be processed is {len(data)}")
label = ""

batch_count = 0

# total_tasks = sum(1 for entry in data if not entry.get("used", False))  
total_tasks = 200
completed_tasks = 0
start_time = time.time()  #


for idx, entry in enumerate(data):
    if not entry.get("used", False):  # Only process data where used is False
        jd = str(entry["jd"])
        print(f"\nProcessing {completed_tasks + 1}/{total_tasks} items...")

        retry_count = 0
        max_retries = 10

        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": """你是一名专业的HR，我将提供给你一份JD和一份标签集，其中标签集是从其他JD中提取出的标签。
                    你的任务是：
                    1. 从JD中提取所有硬技术标签（不包括软技能）。
                    2. 去除所有在标签集中已存在的技术标签，输出剩余的不重复硬技术标签，标签之间用逗号,隔开。
                    3. 如果JD没有任何新的技术标签需要补充，输出 "无"。

                    注意：
                    - 硬技术标签指编程语言、数据库、工具、框架等技术相关内容，不包括软技能。
                    - 如果标签集为空，则直接提取JD中的所有硬技术标签。
                    - 除了规定的输出外，不要输出其他内容。

                    样例：
                    输入：
                        标签集：Python, MySQL
                        JD：精通 Python、熟悉 MySQL 和 Redis，有 Kafka 使用经验
                    输出：
                        Redis,Kafka
                    """},
                        {"role": "user", "content": f"标签集:{label if label else '（空）'},JD:{jd}"}
                    ],
                    stream=False
                )

                answer = response.choices[0].message.content.strip() if response.choices else ""

                if answer:  # Make sure the returned content is not empty
                    print(f"AI-generated labels: {answer}")

                    # Saving API usage
                    usage = response.usage
                    if usage:
                        total_usage["prompt_tokens"] += usage.prompt_tokens
                        total_usage["completion_tokens"] += usage.completion_tokens
                        total_usage["total_tokens"] += usage.total_tokens

                    # Save the results
                    entry["messages"] = {"role": "user", "content": f"标签集:{label},JD:{jd}"}
                    entry["answer"] = answer
                    entry["used"] = True
                    if answer != "无":
                        label += "," + answer if label else answer
                    entry["label"] = label

                    # Save each piece of data after processing to prevent data loss caused by errors
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, ensure_ascii=False, indent=4)

                    batch_count += 1  # counter+1

                    break  # Successfully obtain the answer and jump out of the retry loop
                else:
                    print(f"Try {retry_count + 1}: AI returns null, try again after 5 seconds...")
                    retry_count += 1

            except APIConnectionError as e:
                print(f"Connection error ({completed_tasks + 1}th data, retry {retry_count + 1} times):", e.__cause__)
            except APIError as e:
                print(f"API returned an error ({completed_tasks + 1}th data item, retried {retry_count + 1} times):", e)

            retry_count += 1

        if retry_count == max_retries:
            print(f"Error: The {completed_tasks + 1}th data item failed after {max_retries} retries. The program is exiting.")
            continue

        # Update the number of completed tasks
        completed_tasks += 1

        # Calculate the estimated remaining time
        elapsed_time = time.time() - start_time  
        avg_time_per_task = elapsed_time / completed_tasks  
        remaining_time = avg_time_per_task * (total_tasks - completed_tasks)  

        if completed_tasks >= total_tasks:
            break
        # Format the remaining time (hours:minutes:seconds)
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        seconds = int(remaining_time % 60)

        print(f"Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")

print("All unused data processed!")

# Print API usage after processing is complete
print("\n🔹 Total API usage:")
print(f"  - Total input Token（prompt_tokens）: {total_usage['prompt_tokens']}")
print(f"  - Total output Token（completion_tokens）: {total_usage['completion_tokens']}")
print(f"  - Total Token Count（total_tokens）: {total_usage['total_tokens']}")

balance2 = get_balance()
print("Current CNY balance:", balance2, "Expected Use:", float(balance1) - float(balance2))

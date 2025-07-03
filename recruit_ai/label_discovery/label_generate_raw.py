import json
import asyncio
import time

import transformers
from openai import OpenAI, APIConnectionError, APIError

import get_key

client = OpenAI(api_key=get_key.get_deepseek_key(), base_url="")
# Initialize the tokenizer
chat_tokenizer_dir = "./"
tokenizer = transformers.AutoTokenizer.from_pretrained(
        chat_tokenizer_dir, trust_remote_code=True
        )

# Record the total number of tokens
total_tokens = {"input": 0, "output": 0}
max_tasks = 200  

file_path = "random_jd.json"


async def process_entry(entry, idx, sem, file_lock, progress, token_lock):
    """
    Processes a single data entry, makes an API call, and writes the result to a file upon success, while counting the number of tokens.
    """
    if entry.get("used", False):
        return
    jd = str(entry["jd"])

    # Count the number of input tokens
    input_tokens = len(tokenizer.encode(jd))

    print(f"\nProcessing {progress['completed'] + 1}/{progress['total']}th data item... Input Token: {input_tokens}")

    retry_count = 0
    max_retries = 10

    while retry_count < max_retries:
        try:
            async with sem:  # Limit the number of concurrent connections
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="deepseek-chat",
                    messages=[
                        {"role": "system",
                         "content": "你是一名专业的HR,你的职责是从下面的题目中总结出题目所要求的技术标签，输出格式为你总结出的所有标签，以逗号,隔开，除了规定的输出外不要输出其他内容。样例：python,Mysql,死锁。注意：期望的技术标签为更偏向硬性技术的标签，技术标签不包括性格、个人能力等软技能"},
                        {"role": "user", "content": jd},
                    ],
                    stream=False
                )

            answer = response.choices[0].message.content.strip() if response.choices else ""
            output_tokens = len(tokenizer.encode(answer)) if answer else 0  # Calculate the number of output tokens

            if answer:
                print(f"AI generated label: {answer} (output token: {output_tokens})")

                entry["answer"] = answer
                entry["used"] = True

                # Record the total number of tokens (lock to avoid concurrency issues)
                async with token_lock:
                    total_tokens["input"] += input_tokens
                    total_tokens["output"] += output_tokens

                # Lock files when writing to avoid concurrent write conflicts
                async with file_lock:
                    with open('jd_output2.json', 'w', encoding='utf-8') as file:
                        json.dump(progress['data'], file, ensure_ascii=False, indent=4)
                break  # Successfully obtain the answer and jump out of the retry loop

            else:
                print(f"Try {retry_count + 1}: AI returns null, try again after 5 seconds...")
                retry_count += 1
                await asyncio.sleep(5)

        except Exception as e:
            print(f"Error ({progress['completed'] + 1}th item, retry {retry_count + 1} times): {e}")
            retry_count += 1
            await asyncio.sleep(5)

    if retry_count == max_retries:
        print(f"Error: The {progress['completed'] + 1}th data item failed after {max_retries} retries. The program is exiting.")
        raise Exception("Task Failed")

    # Update the number of completed tasks and calculate the estimated remaining time
    progress['completed'] += 1
    elapsed_time = time.time() - progress['start_time']

    # Terminate after processing max_tasks pieces of data
    if progress['completed'] >= max_tasks:
        print("The required amount of data has been processed, terminating early!")
        return

    avg_time = elapsed_time / progress['completed']
    remaining = avg_time * (progress['total'] - progress['completed'])
    hours, minutes, seconds = int(remaining // 3600), int((remaining % 3600) // 60), int(remaining % 60)
    print(f"Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")


async def main():
    # Reading JSON data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    print("Processing unused data...")

    # Get unprocessed data and limit the maximum number of processed
    unprocessed_entries = [entry for entry in data if not entry.get("used", False)]
    total_tasks = min(len(unprocessed_entries), max_tasks)  # Only process up to max_tasks pieces of data
    print(f"Plan to process {total_tasks} pieces of data (can process at most {max_tasks} pieces)")

    # Defining concurrency control and locks
    sem = asyncio.Semaphore(10)  # Limit the number of concurrent connections
    file_lock = asyncio.Lock()
    token_lock = asyncio.Lock()

    # Progress Information
    progress = {'completed': 0, 'total': max_tasks, 'start_time': time.time(), 'data': data}

    #Creating parallel tasks
    tasks = [process_entry(entry, idx, sem, file_lock, progress, token_lock) for idx, entry in
             enumerate(unprocessed_entries[:max_tasks])]

    await asyncio.gather(*tasks)

    print("Processing completed!")
    print(
        f"Total Input Token: {total_tokens['input']}, Total Output Token: {total_tokens['output']}, Total Token: {total_tokens['input'] + total_tokens['output']}")


if __name__ == '__main__':
    asyncio.run(main())

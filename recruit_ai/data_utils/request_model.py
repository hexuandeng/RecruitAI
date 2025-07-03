import copy
import random
from openai import OpenAI
from transformers import AutoTokenizer
from typing import Dict, List
from recruit_ai.data_utils.prompts import PROMPT_MAP

MODEL_TO_PORT = {
    "Qwen2-7B-Instruct": ("base", 20071),
    "Qwen2.5-72B-Instruct": ("model", 25723),
    "Label": ("label", 20071),
    "ChatGLM": ("model", 30070),
    "Deepseek": ("model", 30071),
}
tokenizer = AutoTokenizer.from_pretrained("Qwen2-7B-Instruct")

def construct_message(task: str, **kwargs) -> List[Dict[str, str]]:
    messages = [{
        'role':
        'system',
        'content':
        PROMPT_MAP[task]['system_prompt'](kwargs)
        if task.startswith('classify') or task.startswith("question_generate") else PROMPT_MAP[task]['system_prompt']
    }]
    if 'few_shot_prompts' in PROMPT_MAP[task]:
        for pair in PROMPT_MAP[task]['few_shot_prompts']:
            messages += [{
                'role': 'user',
                'content': pair['user']
            }, {
                'role': 'assistant',
                'content': pair['assistant']
            }]
    messages.append({
        'role': 'user',
        'content': PROMPT_MAP[task]['user_prompt'](kwargs)
    })
    return messages


def truncate_message_content(message, tokenizer, max_tokens=2048):
    """
    Truncate the content of a single message to ensure that the number of tokens does not exceed max_tokens.
    The latter part of the content is truncated, and the former part is retained.
    """
    content = message.get("content", "")
    tokens = tokenizer.encode(content, add_special_tokens=False)

    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_content = tokenizer.decode(truncated_tokens,
                                             skip_special_tokens=True)
        message["content"] = truncated_content

    return message


def truncate_messages(messages, tokenizer, max_tokens=2048):
    """
    Truncate each message in the messages list to ensure that the number of tokens in each message does not exceed max_tokens.
    """
    truncated_messages = []
    for message in messages:
        truncated_message = truncate_message_content(copy.deepcopy(message),
                                                     tokenizer, max_tokens)
        truncated_messages.append(truncated_message)
    return truncated_messages


def chat(messages, model, **kwargs):
    # Modify OpenAI's API key and API base to use vLLM's API server.
    # https://platform.openai.com/docs/api-reference/chat/create
    messages = truncate_messages(messages, tokenizer, max_tokens=2048)

    if "temperature" not in kwargs:
        kwargs["temperature"] = 0

    if "gpt" in model:
        client = OpenAI(
            api_key="",
            base_url="",
        )
        completion = client.chat.completions.create(messages=messages,
                                                    stream=False,
                                                    model=model,
                                                    **kwargs)
    else:
        openai_api_key = "EMPTY"
        port = MODEL_TO_PORT[model][1]
        if isinstance(port, list):
            port = random.choice(port)
        openai_api_base = f"http://localhost:{port}/v1"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        # Completion API
        completion = client.chat.completions.create(
            messages=messages,
            stream=False,
            model=MODEL_TO_PORT[model][0],
            **kwargs)
    #print(completion.usage)
    #print(completion)
    return completion.choices[0].message.content.strip()


def chat_prob(messages, model, **kwargs):
    # Modify OpenAI's API key and API base to use vLLM's API server.
    # https://platform.openai.com/docs/api-reference/chat/create
    messages = truncate_messages(messages, tokenizer, max_tokens=2048)

    if "temperature" not in kwargs:
        kwargs["temperature"] = 0

    if "gpt" in model:
        client = OpenAI(
            api_key="",
            base_url="",
        )
        completion = client.chat.completions.create(messages=messages,
                                                    stream=False,
                                                    model=model,
                                                    **kwargs)
    else:
        openai_api_key = "EMPTY"
        port = MODEL_TO_PORT[model][1]
        if isinstance(port, list):
            port = random.choice(port)
        openai_api_base = f"http://localhost:{port}/v1"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        # Completion API
        completion = client.chat.completions.create(
            messages=messages,
            stream=False,
            model=MODEL_TO_PORT[model][0],
            logprobs=True,
            top_logprobs=5,
            **kwargs)

    return completion.choices[0].logprobs.content[0].top_logprobs


def request_model(model_name: str, task: str, **kwargs) -> str:
    messages = construct_message(task, **kwargs)
    # print(messages)
    return chat(messages, model_name)

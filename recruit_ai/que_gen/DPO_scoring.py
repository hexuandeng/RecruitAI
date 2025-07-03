import argparse
import torch
import vllm
import math
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Reject Sample Task with VLLM")
    parser.add_argument("--gen_model_path", type=str, required=True, help="Generate model path")
    parser.add_argument("--ppl_model_path", type=str, required=True, help="Model path for calculating PPL")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num_replies", type=int, default=5, help="The number of responses generated per prompt")
    parser.add_argument("--max_tokens", type=int, default=2048, help="The maximum number of tokens generated")
    parser.add_argument("--batch_size", type=int, default=8, help="Processing batch size")
    parser.add_argument("--stop_sequences", type=str, nargs='*', default=["\n\n"], help="Generated stop sequence")
    parser.add_argument("--input_file", type=str, required=True, help="Enter the file path of the prompt, one prompt per line")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode and process only part of the data")
    parser.add_argument('--length-norm', action='store_true', help='Enable length normalization when selecting the best reply')
    parser.add_argument('--length-weight', type=float, default=0.1, help='Weight for length normalization when selecting the best reply')
    parser.add_argument("--output_file", type=str, required=True, help="The file path where the generated responses and their PPL scores are saved")
    return parser.parse_args()

def generate_replies(model, sampling_params, prompts, num_replies):
    """
    Generates the specified number of responses for all prompts.

    Args:
        model: VLLM Generate Model.
        sampling_params: Sampling parameters.
        prompts: Enter a prompt list.
        num_replies: The number of responses generated per prompt.

    Returns:
        List[str]: List of all generated responses.
    """
    # Expand each prompt to num_replies identical prompts
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * num_replies)
    
    # Generate all replies
    generations = model.generate(expanded_prompts, sampling_params)
    
    # Extract reply text
    all_replies = [g.outputs[0].text for g in generations]
    
    return all_replies

def extract_first_question_answer(prompts,replies,num_replies):
    """
    Extract the contents of the first pair of "Question:" and "Answer:" from the string array.
    
    Args:
        replies (list of str): An array containing multiple strings, each of which may contain one or more pairs of "Question:" and "Answer:".
    
    Returns:
        list of str: Extract the string array after the first pair of "Question:" and "Answer:".
    """

    extracted_replies = []
    for idx,reply in enumerate(replies):
        if prompts[idx // num_replies].startswith("你是一个文本处理助手。你的任务是根据用户提供的无标注文本，生成一道选择题"):
            match = re.search(r'题目：(.*?)答案：(.*?)(?=\n题目：|$)', reply, re.DOTALL)
            if match:
                # Extract the content of the first pair of "question" and "answer"
                question = match.group(1).strip()
                answer = match.group(2).strip().split("\n")[0]
                extracted_replies.append(f"问题：{question} 回答：{answer}")
            else:
                # If no match is found, keep the original string.
                extracted_replies.append(reply)
        elif prompts[idx // num_replies].startswith("你是一个文本处理助手。你的任务是根据用户提供的无标注文本，生成一道简答题"):
            match = re.search(r'问题：(.*?)回答：(.*?)(?=\n问题：|$)', reply, re.DOTALL)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip().split("\n")[0]
                extracted_replies.append(f"问题：{question} 回答：{answer}")
            else:
                extracted_replies.append(reply)
        else:  # Fill-in-the-blank and True-or-False questions
            match = re.search(r'题目：(.*?)答案：(.*?)(?=\n题目：|$)', reply, re.DOTALL)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip().split("\n")[0]
                extracted_replies.append(f"问题：{question} 回答：{answer}")
            else:
                extracted_replies.append(reply)
        # Use a regular expression to match the first pair of "Question:" and "Answer:"
        
    return extracted_replies

def compute_ppl(tokenizer, model, replies, batch_size=8):
    """
    Compute the perplexity of a set of responses (PPL).

    Args:
        tokenizer: The tokenizer to use for encoding.
        model: Transformers model for computing PPL.
        replies: List of replies for which PPL needs to be calculated.
        batch_size: Batch size.

    Returns:
        List[float]: The PPL score corresponding to each response.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ppls = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(replies), batch_size), desc="Calculating PPL"):
            batch = replies[i:i+batch_size]
            # Tokenize the response
            encodings = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            labels = input_ids.clone()

            # Computational model output
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            # Calculate the cross entropy loss for each token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            # Calculate the average loss of each sample based on the attention mask
            loss_per_sample = (loss * attention_mask[:,1:]).sum(dim=1) / attention_mask[:,1:].sum(dim=1)

            loss_per_sample = loss_per_sample.cpu().tolist()
             # Replace all NaNs with a very large value (ensuring they don't get picked as lowest PPL)
            loss_per_sample = [float('inf') if math.isnan(x) else x for x in loss_per_sample]

            # Calculate PPL and save
            ppls.extend([torch.exp(torch.tensor(l)).item() for l in loss_per_sample])
    return ppls

# Find the best response that takes into account both PPL and length regularization
def calculate_score(reply, args, min_ppl, max_ppl, min_length, max_length):
    ppl = reply['ppl']
    length = reply['length']
    if args.length_norm:
        # Normalize PPL and length to ensure a balanced weight between the two
        normalized_ppl = (ppl - min_ppl) / (max_ppl - min_ppl) if max_ppl > min_ppl else 0
        normalized_length = (length - min_length) / (max_length - min_length) if max_length > min_length else 0
        return normalized_ppl - args.length_weight * normalized_length  # Use the passed in weight parameter
    return ppl  # If length regularization is not enabled, only PPL is considered


def main():
    args = parse_args()
    
    print("Loading generation model...")
    gen_model = vllm.LLM(
        model=args.gen_model_path,
        tokenizer=args.gen_model_path,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.95,
        max_model_len=4096
    )
    print("Generation model loaded.")

    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=0.95 if args.temperature != 0 else None,
        max_tokens=args.max_tokens,
        stop=args.stop_sequences,
    )
    
    print("Reading input prompts...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    
    if args.debug:
        datas = datas[200:210]
        args.output_file = "./debug.json"
    
    prompts = [item["prompt"] for item in datas]
    print(f"Total prompts loaded: {len(prompts)}")

    print("Generating replies...")
    all_replies = generate_replies(gen_model, sampling_params, prompts, args.num_replies)
    print("Replies generated.")

    all_replies = extract_first_question_answer(prompts,all_replies, args.num_replies)

    print("Releasing generation model resources...")
    del gen_model
    torch.cuda.empty_cache()
    print("Generation model resources released.")

    print("Loading PPL model...")
    ppl_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model_path)
    ppl_model = AutoModelForCausalLM.from_pretrained(args.ppl_model_path)
    ppl_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppl_model.to(device)
    print("PPL model loaded.")

    print("Computing PPL for all replies...")
    ppls = compute_ppl(ppl_tokenizer, ppl_model, all_replies, batch_size=args.batch_size)
    print("PPL computation completed.")

    print("Organizing results...")
    results = []
    idx = 0
    for item, prompt in zip(datas, prompts):
        generated_replies = []
        for _ in range(args.num_replies):
            reply = all_replies[idx]
            ppl_score = ppls[idx]
            match = re.search(r'问题：(.*?)回答：(.*?)(?=\n问题：|$)', reply, re.DOTALL)
            if match:
                #print("====match=====")
                question = match.group(1).strip()  
                answer = match.group(2).strip()    
            else:
                question, answer = reply.split("回答：", 1) if "回答：" in reply else (reply, "")
            # generated_replies.append({'question': question.strip(), 'answer': answer.strip(), 'ppl': ppl_score})
            generated_replies.append({'question': question.strip(), 'answer': answer.strip(), 'ppl': ppl_score, 'length': len(answer.strip())})
            idx += 1
        # Find the lowest response from PPL
        # Get the maximum and minimum values ​​of PPL and length for normalization
        min_ppl = min(reply['ppl'] for reply in generated_replies)
        max_ppl = max(reply['ppl'] for reply in generated_replies)
        min_length = min(reply['length'] for reply in generated_replies)
        max_length = max(reply['length'] for reply in generated_replies)

        best_reply = min(generated_replies, key=lambda reply: calculate_score(reply, args, min_ppl, max_ppl, min_length, max_length))
        # best_reply = min(generated_replies, key=lambda x: x['ppl'])
        item["all_model_generate"] = generated_replies
        item["model_generate"] = best_reply
        results.append(item)

    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Results saved successfully.")

if __name__ == "__main__":
    main()

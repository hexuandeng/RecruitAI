from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import torch
from tqdm import tqdm
SYSTEM_PROMPT=("现在我给你一个岗位名称，请你在以下行业大类选项中选择一项与之最为契合的选项。"
                   "可选项为：A.市场营销 B.产品/项目/运营 C.通信/硬件 D.咨询/管理 E.人力/财务/行政"
                   "F.供应链/物流 G.机械/制造 H.视觉/交互/设计 I.金融 J.软件开发 K.教育/科研 L.生物医药"
                   "只需输出选项即可。")

options = ["A.市场营销", "B.产品/项目/运营", "C.通信/硬件", "D.咨询/管理", "E.人力/财务/行政", 
           "F.供应链/物流","G.机械/制造","H.视觉/交互/设计","I.金融","J.软件开发","K.教育/科研",
           "L.生物医药"]


def load_model_tokenizer(model_name_or_path):
    print('loading model...')
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 torch_dtype="auto",
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    
    return model,tokenizer

def prepare_inputs(tokenizer, texts):
    # Tokenize a list of texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Send all input data to the default GPU device
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items() if torch.is_tensor(v)}
    
    return inputs


def main(args):
    
    model,tokenizer = load_model_tokenizer(args.model_name_or_path)
    with open(args.input_file,"r",encoding="utf-8") as f:
        job_name_datas = json.load(f)
    
    if args.debug:
        args.output_file = "./debug.json"
        job_name_datas = job_name_datas[:400]

    option_ids = [tokenizer.encode(opt)[0] for opt in options]

    # Filter it first.
    print(f"Before filtration：{len(job_name_datas)}")
    job_name_datas = [data for data in job_name_datas if data[args.key_name] != None]
    print(f"After filtering：{len(job_name_datas)}")


    data_num = len(job_name_datas)
    split_num = data_num // 4
    if args.part is not None:
        if args.part < 3:
            job_name_datas = job_name_datas[args.part *
                                            split_num:(args.part + 1) *
                                            split_num - 1]
        else:
            job_name_datas = job_name_datas[3 * split_num:]
        args.output_file = args.output_file[:-5] + '_part' + str(
            args.part) + '.json'

    # batch size processing
    batch_size = args.batch_size
    results = []
    for i in tqdm(range(0, len(job_name_datas), batch_size)):
        batch = job_name_datas[i:i + batch_size]
        texts = ["岗位名称: " + data[args.key_name] + "\n\n" for data in batch]
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT} for _ in batch]
        user_messages = [{"role": "user", "content": text} for text in texts]

        chat_messages = [[m,u] for m, u in zip(messages, user_messages)]

        
        formatted_texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in chat_messages]

        inputs = prepare_inputs(tokenizer, formatted_texts)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.detach()
        logits = logits[:,-1,:] # logits of the last token, [batch_size,seq_len, vocab_size]

        # Handle different batch sizes
        for j in range(len(batch)):
            logits_options = logits[j, option_ids]
            probabilities = torch.softmax(logits_options, dim=0)
            max_index = torch.argmax(probabilities)
            batch[j]["所属大类"] = options[max_index][2:]  
            results.append(batch[j])


    # Save or process results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path',
                        type=str,
                        default='Qwen2-72B-Instruct')
    parser.add_argument('--input_file',
                        type=str,
                        default='需求数据.json')
    parser.add_argument('--output_file',
                        type=str,
                        default='./岗位_大类.json')
    parser.add_argument("--key_name",type=str,default="需求岗位")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--part', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)

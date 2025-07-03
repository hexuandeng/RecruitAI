import os
import time
import torch
import random
from multiprocessing import Process, Queue, Value, Lock
from transformers import AutoModelForCausalLM, AutoTokenizer
from queue import Empty, Full
from transformers import GenerationConfig
from transformers import LlamaTokenizer, LlamaTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = ""
MAX_TOKEN = 2048

JOBCATEGORY = [
    "市场营销", "产品/项目/运营", "通信/硬件", "咨询/管理", "人力/财务/行政", "供应链/物流", "机械/制造",
    "视觉/交互/设计", "金融", "软件开发", "教育/科研", "生物医药"
]

SYSTEM_PROMPTS = [
    "我将给你提供一个技术标签，请你从下列行业大类中选择一项与之最为契合的选项，如果均不契合请选择NOT FOUND。",
    "我将给你提供一个工作岗位及其对应的岗位职责等，请你从下列行业大类中选择一项与之最为契合的选项。",
    "我将给你提供一个工作岗位及其对应的岗位职责等，请你从下列技术标签中选择一项在面试时该岗位需要考察的技术点，如果均无需考察请选择NOT FOUND。",
    "我将给你提供一个工作岗位对应的工作描述，请你从下列行业大类中选择一项与之最为契合的选项。",
    "我将给你提供一个工作岗位对应的工作描述，请你从下列技术标签中选择一项与之最为契合的选项，如果均不契合请选择NOT FOUND。",
    "我将给你提供一个工作项目经历对应的描述，以及在这个项目中负责的职责，请你从下列行业大类中选择一项与之最为契合的选项，如果均不契合请选择NOT FOUND。",
    "我将给你提供一个工作项目经历对应的描述，以及在这个项目中负责的职责，请你从下列技术标签中选择一项与之最为契合的选项，如果均不契合请选择NOT FOUND。",
    "我将给你提供一道题目及其正确答案，请你从下列行业大类中选择一项与之最为契合的选项。",
    "我将给你提供一道题目及其正确答案，请你从下列技术标签中选择一项与之最为契合的选项，如果均不契合请选择NOT FOUND。",
    "你是一名专业的HR。现在我将给你提供一个技术标签，请你根据我提供的工作描述这项简历信息来判断该技术标签是否属于工作描述这项简历信息的考察范围。可选项为：是 否。请注意你只需要回答是/否即可。",
    "你是一名专业的HR。现在我给你提供一个技术标签，请你根据我提供的岗位职责以及任职要求这两项JD信息来判断该技术标签是否属于岗位职责以及任职要求这两项JD信息的考察范围。可选项为：是 否。请注意你只需要回答是/否即可。",
    "你是一名专业的HR。现在我给你提供一个技术标签，请你根据我提供的工作项目经历对应的描述以及在这个项目中负责的职责这两项简历信息来判断该技术标签是否属于项目描述及项目职责这两项简历信息的考察范围。可选项为：是 否。请注意你只需要回答是/否即可。",
    "你是一名专业的HR。现在我将给你提供一个岗位的工作描述，以及一个行业大类。请根据所给的描述判断该岗位是否属于所给行业大类的范畴。可选项为：是 否。请注意，你只需回答是/否即可。",
    "你是一名专业的HR。我将提供一个岗位的岗位职责，该岗位的任职要求，以及一个行业大类。请根据岗位职责判断任职要求及岗位职责是否属于所给行业大类的范畴。可选项为‘是’或‘否’。请注意，你只需回答‘是’或‘否’即可。",
    "你是一名专业的HR。我将提供一个工作项目经历的项目描述、该项目中负责的职责，以及一个行业大类。请根据项目职责判断项目描述及项目职责是否属于所给行业大类的范畴。可选项为‘是’或‘否’。请注意，你只需回答‘是’或‘否’即可，并且在选择时应重点关注项目中担任的职责，而非项目的整体描述。"
]


def load_tokenizer(model_path):
    if "Qwen" in model_path or "internlm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
    elif "SUS" in model_path or "Yi-34B-Chat" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "left"
    # tokenizer.model_max_length = min(tokenizer.model_max_length, 4096)
    return tokenizer

def load_model_tokenizer(model_path):
    if "Qwen" in model_path or "internlm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
    elif "SUS" in model_path or "Yi-34B-Chat" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = min(tokenizer.model_max_length, 4096)

    if "falcon" in model_path or "deepseek" in model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True)
        if "deepseek" in model_path:
            model.generation_config = GenerationConfig.from_pretrained(
                model_path)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif "Mistral" in model_path or "mpt" in model_path or "Mixtral" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto")
    elif "Qwen" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     trust_remote_code=True)
    elif "internlm" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
    elif "COKAL" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.float16,
                                                     return_dict=True)
    elif "glm" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.float16)

    model.eval()
    return model, tokenizer


def get_results_from_queue(tokenizer, queue, buffer, max_words, timeout=1):
    to_process = []
    len_tok = []
    retry = 3
    while not len(to_process) or \
        min(max(len_tok), MAX_TOKEN) * (len(to_process) + 1) < max_words:
        try:
            if not buffer.empty():
                result = buffer.get()
            else:
                result = queue.get(timeout=timeout)
            to_process.append(result)
            len_tok.append(len(tokenizer(result[0])["input_ids"]))
            if len(to_process) > 1 and \
                min(max(len_tok), MAX_TOKEN) * len(to_process) > max_words:
                buffer.put(to_process.pop(-1))
                len_tok.pop(-1)
                retry -= 1
                if retry == 0:
                    break
        except Empty:
            break

    # print(len_tok)
    return to_process


def prepare_inputs(tokenizer, texts):
    # Tokenize a list of texts
    inputs = tokenizer(texts,
                       max_length=MAX_TOKEN,
                       return_tensors="pt",
                       padding=True,
                       truncation=True)
    if torch.cuda.is_available():
        inputs = {
            k: v.to("cuda")
            for k, v in inputs.items() if torch.is_tensor(v)
        }
    return inputs


class MultiInference():

    def __init__(self, args, n_gpus, gpu_per_process=1, int4=True) -> None:
        # n_gpus is the total number of GPUs in the server. Currently, the implementation cannot call torch before the process starts, so torch detection cannot be used here.
        # gpu_per_process is the number of GPUs used by each process
        self.write_file = getattr(args, "output_file", None)
        self.model_name_or_path = args.model_name_or_path
        self.max_words = args.max_words
        self.gpu_per_process = gpu_per_process
        self.n_gpus = n_gpus
        self.int4 = int4
        self.processes = len(n_gpus) // gpu_per_process
        self.input_queue = Queue(maxsize=100)
        self.output_queue = Queue(maxsize=100)
        self.buffer = Queue()
        self.process_list = []
        self.processing = Value('i', 0)
        self.processing_Lock = Lock()

    def forward(self, world_id):
        # Automatically returns the next logits
        gpus = [
            i + world_id * self.gpu_per_process
            for i in range(self.gpu_per_process)
        ]
        gpus = [self.n_gpus[i] for i in gpus]
        gpus = ",".join(map(str, gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print("Rank", world_id, "Loading Model on GPU", gpus,
              torch.cuda.device_count())

        if self.int4:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                quantization_config={"load_in_4bit": True},
                device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto")
        print(f'Rank {world_id} loading model finished!')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side= "right"

        # model,tokenizer = load_model_tokenizer(self.model_name_or_path)

        while True:
            to_process = get_results_from_queue(tokenizer, self.input_queue,
                                                self.buffer, self.max_words)
            print(to_process)
            if len(to_process) == 0:
                time.sleep(1)
                continue
            datas = None
            if len(to_process[0]) == 2:
                datas = [i[1] for i in to_process]
                to_process = [i[0] for i in to_process]
            inputs = prepare_inputs(tokenizer, to_process)
            with torch.inference_mode():
                outputs = model(**inputs)

            logits = outputs.logits.detach()
            logits = logits[:,
                            -1, :]  # logits of the last token, [batch_size,seq_len, vocab_size]
            for i in range(logits.shape[0]):
                self.output_queue.put(
                    (logits[i], datas[i] if datas is not None else None))

    def start(self, post_processing=None):
        # Start all processes. Do not call torch before this, otherwise an error will be reported later.
        # post_processing is the post-processing function that needs to be performed after getting the model output
        # The input is the model output and the data passed in by the push function; the output is the string that needs to be written to the file
        assert len(self.process_list) == 0, "Process already started"
        for i in range(self.processes):
            p = Process(target=self.forward, args=(i, ))
            self.process_list.append(p)
            p.start()
        self.write_process = Process(target=self.write,
                                     args=(post_processing, ))
        self.write_process.start()

    def wait_finish(self):
        # Block until all queued requests have completed
        while self.processing.value > 0:
            print(f"Waiting for {self.processing.value} tasks to finish")
            time.sleep(1)
        for p in self.process_list:
            p.terminate()
        self.process_list = []
        self.write_process.terminate()

    def push(self, string):
        # Push the content that needs to be processed into the queue
        # It is recommended to push tuple (string, data)
        # string is the input string to be processed
        # Data is other information that needs to be written together with the result and will be passed to the post_processing function
        self.input_queue.put(string)
        with self.processing_Lock:
            self.processing.value += 1

    def write(self, post_processing=None):
        with open(self.write_file, 'w') as f:
            pass
        while True:
            logit, data = self.output_queue.get()
            with open(self.write_file, 'a', encoding='utf-8') as w:
                s = post_processing(logit, data)
                w.write(s.strip() + '\n')
                with self.processing_Lock:
                    self.processing.value -= 1


def post_processing_max(logits, data):
    option_ids = data["option_ids"]
    labels = data["labels"]

    logits_options = logits[option_ids]
    probabilities = torch.softmax(logits_options, dim=0)
    max_index = torch.argmax(probabilities)

    # if random.randint(0, 100) < 1:
    #     print(data["text"])
    #     for i, k in zip(labels, probabilities):
    #         print(i, k.item())
    #     print()

    return (data["query_id"], [(labels[max_index],
                                probabilities[max_index].item())])


def post_processing_multi(logits, data):
    discard_last = False

    option_ids = data["option_ids"]
    labels = data["labels"]
    threshold = 0.1
    if labels[-1] == 'NOT FOUND':
        discard_last = True
    if len(labels) == 2:
        threshold = 0

    logits_options = logits[option_ids]
    probabilities = torch.softmax(logits_options, dim=0)

    # if random.randint(0, 100) < 1:
    #     print(data["text"])
    #     for cnt, (i, k) in enumerate(zip(labels, probabilities)):
    #         # print(i, k.item(), torch.softmax(torch.tensor([logits_options[cnt], logits_options[-1]]), dim=0)[0].item())
    #         print(i, k.item())
    #     print()

    it = zip(labels, probabilities)
    if discard_last:
        it = zip(labels[:-1], probabilities[:-1])

    results = []  
    max_pro = 0
    for i, k in it:
        if k > max_pro :
            max_index = i
            max_pro = k
        if k > threshold:# or i == 'NOT FOUND':
            results.append((i, k.item()))
            
    # if len(results) == 0:           
    #     results.append((max_index, max_pro.item()))
    return (data["query_id"], results)


def post_processing_two(logits, data):
    option_ids = data["option_ids"]
    labels = data["labels"]

    logits_options = logits[option_ids]
    probabilities = torch.softmax(logits_options, dim=0)

    return (data["query_id"], [(labels[0],
                                probabilities[0].item())])


class LabelInference(MultiInference):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        # self.tokenizer.padding_side = "left"
        self.tokenizer = load_tokenizer(self.model_name_or_path)
        print(self.tokenizer)
        uppercase_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        self.option_ids = [
                self.tokenizer.encode(opt, add_special_tokens=False)[0] for opt in uppercase_letters
            ]
            
        print(self.option_ids)
        self.all_results = {}

        self.final_queue = Queue()
        self.final_Lock = Lock()
        self.post_processing = post_processing_multi
        self.start()

    def push(self, ipt_str, labels, prompt_id, query_id):
        '''
        ipt_str: Pending User PROMPT
        labels：Label Set
        prompt_id：The prompt id used
        query_id：Unique id for each request, used for get_by_index to get results
        '''
        # print(f'push: {ipt_str}\n\n')
        data = {}

        def add_options(labels):
            options = []
            for index, label in enumerate(labels):
                option = chr(ord('A') + index) + '. ' + label
                options.append(option)
            return options

        if len(labels) == 2 and "是" in labels and "否" in labels:
            SYSTEM_PROMPT = SYSTEM_PROMPTS[prompt_id]
            option_ids = [self.tokenizer.encode(opt,add_special_tokens=False)[0] for opt in ["是", "否"]]
            self.post_processing = post_processing_two
        else:
            SYSTEM_PROMPT = f"你是一名专业的HR。{SYSTEM_PROMPTS[prompt_id]}" + \
                            " ".join(add_options(labels)) + "\n" + \
                            "请从以上选项中选择一个，并输出对应的字母选项。注意只需输出选项即可。"
            option_ids = self.option_ids[:len(labels)]

        messages = [{
            "role": "system",
            "content": SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": ipt_str
        }]
        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        
        data["text"] = ipt_str
        data["option_ids"] = option_ids
        data["labels"] = labels
        data["prompt_id"] = prompt_id
        data["query_id"] = query_id

        super().push((text, data))

    def get_by_index(self, index):
        while True:
            # with self.final_Lock:
            while not self.final_queue.empty():
                it = self.final_queue.get()
                self.all_results[it[0]] = it[1]
            if index in self.all_results:
                return self.all_results.pop(index)
            if random.randint(0, 10) < 1:
                print("Waiting", index)
            time.sleep(1)

    def write(self, post_processing=None):
        while True:
            logit, data = self.output_queue.get()
            # if data["prompt_id"] in [0, 2]:
            #     post_processing = post_processing_max
            # else:
            post_processing = self.post_processing
            s = post_processing(logit, data)
            # print(s)

            self.final_queue.put(s)
            with self.processing_Lock:
                self.processing.value -= 1
            # print(self.processing.value)

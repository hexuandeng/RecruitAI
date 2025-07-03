import argparse
from typing import Dict, List

from flask import Flask, jsonify, request
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, SamplingParams

app = Flask(__name__)


class FillInEvaluator:

    SYSTEM_PROMPT = "你是一名资深的评价员。我将提供一道填空题及其正确答案和候选人给出的答案，请你根据正确答案及候选人给出的答案给候选人打分，注意不要输出除分数外多余部分。"

    def __init__(self, llm: LLM, tokenizer: PreTrainedTokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    def construct_message(
        self,
        fill_in_qa: Dict[str, str],
        candidate_answer: str,
    ) -> List[Dict[str, str]]:
        return [{
            "role":
            "system",
            "content":
            "你是一名资深的评价员。我将提供一道填空题及其正确答案和候选人给出的答案，请你根据正确答案及候选人给出的答案给候选人打分，注意不要输出除分数外多余部分。"
        }, {
            "role":
            "user",
            "content":
            "### 问题：请列举出JVM的四个主要组成部分。\n\n### 正确答案：类加载器子系统、运行时数据区、执行引擎、本地接口库\n\n### 候选人答案：类加载器子系统、运行时数据区、执行引擎、本地接口库\n\n请你根据正确答案对候选人答案进行打分，满分10分，注意不要输出分数以外多余部分。Assistant: "
        }, {
            "role": "assistant",
            "content": "6"
        }, {
            "role":
            "user",
            "content":
            "### 问题：在Java中，如何手动触发垃圾回收？\n\n### 正确答案：通过调用 System.gc() 或 Runtime.getRuntime().gc()\n\n### 候选人答案：通过调用 system.gc() \n\n请你对此进行打分，满分10分，注意不要输出分数以外多余部分。Assistant: "
        }, {
            "role": "assistant",
            "content": "2"
        }, {
            "role":
            "user",
            "content":
            f"### 问题：{fill_in_qa['question']}\n\n### 正确答案：{fill_in_qa['answer']}\n\n### 候选人答案：{candidate_answer}\n\n请你对此进行打分，满分10分，注意不要输出分数以外多余部分。Assistant: "
        }]

    def generate(
        self,
        fill_in_qa: Dict[str, str],
        candidate_answer: str,
    ) -> str:
        message = self.construct_message(fill_in_qa, candidate_answer)
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.llm.generate([text], self.sampling_params)
        response = outputs[0].outputs[0].text
        return response


def get_match_score(
    jd: Dict[str, int | str],
    score_result: Dict[str, str],
) -> float:
    match_sum = 0
    num = 0
    labels = [
        label_score["label"] for label_score in score_result["label_scores"]
    ]
    label_score_dic = {
        label_score["label"]: label_score["score"]
        for label_score in score_result["label_scores"]
    }
    for label in jd['label']:
        if label in labels:
            match_sum += label_score_dic[label]
            num += 1
    return 0 if num == 0 else match_sum / num


def get_max_label_score(jd, score_result):
    max_label_socre = 0
    max_label = ""
    labels = [
        label_score["label"] for label_score in score_result["label_scores"]
    ]
    label_score_dic = {
        label_score["label"]: label_score["score"]
        for label_score in score_result["label_scores"]
    }
    for label in jd['label']:
        if label in labels and label_score_dic[label] > max_label_socre:
            max_label_socre = label_score_dic[label]
            max_label = label
    return max_label, max_label_socre


def llm_generate(message: List[Dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate([text], sampling_params)
    response = outputs[0].outputs[0].text
    return response


def construct_eval_highest_score_label_message(
    highest_score_label: str, ) -> List[Dict[str, str]]:
    return [{
        "role":
        "system",
        "content":
        "你是一名资深的评价员。我将提供一份答卷中得分率最高的标签及其得分，满分100分，请你根据该标签及其得分给出分析。"
    }, {
        "role": "user",
        "content": "候选人得分最高的标签为网络基础，得分为100。请你对此进行分析，注意不要输出分析外多余部分。"
    }, {
        "role":
        "assistant",
        "content":
        "得分最高的标签为“网络基础”，得分为100分，这表明在评估的范围内，该候选人在网络基础方面表现出了极高的专业水平和知识掌握程度。"
    }, {
        "role": "user",
        "content": "候选人得分最高的标签为Redis，得分为89。请你对此进行分析，注意不要输出分析外多余部分。"
    }, {
        "role":
        "assistant",
        "content":
        "Redis是一个非常受欢迎的键值储存系统，尤其在构建高性能、低延迟的应用场景中表现突出。得分89分表明候选人对Redis有深入的理解和熟练的应用能力，这在现代软件开发中是非常有价值的技能。"
    }, {
        "role":
        "user",
        "content":
        f"候选人得分最高的标签为{highest_score_label['label']}，得分为{highest_score_label['score']}。请你对此进行分析，注意不要输出分析外多余部分。Assistant: "
    }]


def construct_eval_lowest_score_label_message(
    lowest_score_label: str, ) -> List[Dict[str, str]]:
    return [{
        "role":
        "system",
        "content":
        "你是一名资深的评价员。我将提供一份答卷中得分率最低的标签及其得分，满分100分，请你根据该标签及其得分给出分析。提出改进建议。"
    }, {
        "role": "user",
        "content": "候选人得分最低的标签为项目管理，得分为1。请你对此进行分析，注意不要输出分析外多余部分。"
    }, {
        "role":
        "assistant",
        "content":
        "得分最低的标签“项目管理”得分为1，这表明候选人在这方面的知识、技能和经验明显不足。项目管理是确保项目成功的关键因素之一，涉及规划、执行、监控和控制项目，以实现既定的目标、时间表和预算。"
    }, {
        "role": "user",
        "content": "候选人得分最低的标签为证券基础，得分为15。请你对此进行分析，注意不要输出分析外多余部分。"
    }, {
        "role":
        "assistant",
        "content":
        "得分最低的标签证券基础得分为15分，这表明在该领域，候选人对证券市场的基本概念、投资工具、交易规则、风险管理和投资策略等方面的知识掌握程度较低。"
    }, {
        "role":
        "user",
        "content":
        f"候选人得分最低的标签为{lowest_score_label['label']}，得分为{lowest_score_label['score']}。请你对此进行分析，注意不要输出分析外多余部分，输出完整句子。Assistant: "
    }]


@app.route('/evaluate_fill_in', methods=['POST'])
def evaluate_fill_in():
    json_data = request.get_json()
    try:
        print(f'request data: {json_data}')
        response = scorer.generate(json_data['fill_in_qa'],
                                   json_data['candidate_answer'])
        json_data = {'response': response}
        print(f'response data: {json_data}')
        return jsonify(json_data), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/normal_generate', methods=['POST'])
def normal_generate():
    json_data = request.get_json()
    try:
        print(f'request data: {json_data}')
        response = llm_generate(json_data['message'])
        json_data = {'response': response}
        print(f'response data: {json_data}')
        return jsonify(json_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main(args: argparse.Namespace):
    global llm, tokenizer, sampling_params
    llm = LLM(model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    print('initializing scorer...')
    global scorer
    scorer = FillInEvaluator(llm, tokenizer)
    print(f'Start server on port {args.port}')
    app.run(port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default=
        'Qwen2-7B-Instruct')
    parser.add_argument('--port', type=int, default=8081)
    main(parser.parse_args())

from openai import OpenAI
import os
import json
import argparse
import re
import tqdm
api_base ="" #chat_gpt base
API_KEY = "" # chat_gpt api
MODLE_TYPE = "gpt-4o-mini"   #gpt-3.5-turbo
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=API_KEY,
    base_url=api_base,
)
def get_system_prompt(prompt_type):
    prompts = {
        1: "现在我要提供你一份真实标签以及一份模型预测标签，请你对模型预测标签与真实标签之间的契合度打个分。分值要求: 1~5分。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n",
        2: "现在我要提供你一份真实标签以及一份模型预测标签，这两份标签都是根据岗位职责和任职要求标注出来的标签。请你结合岗位职责和任职要求对模型预测标签与真实标签之间的契合度打个分，分值要求: 1~5分，1分为所有标签不相关，5分为所有标签都相关。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n",
        3: "现在我要提供你一份真实标签以及一份模型预测标签，这两份标签是根据工作描述信息，由模型标注出来的标签。请你结合工作描述信息对模型预测标签与真实标签之间的契合度打个分，分值要求: 1~5分，1分为所有标签不相关，5分为所有标签都相关。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n",
        4: "现在我要提供你一份模型预测标签，这份标签是根据岗位名称、岗位职责和任职要求信息，由模型标注出来的标签。请你结合岗位名称、岗位职责和任职要求信息和模型预测标签之间的契合度打个分，分值要求: 1~5分，1分为所有标签不相关，5分为所有标签都相关。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n",
        5: "现在我要提供你一份模型预测标签，这份标签是根据工作描述信息，由模型标注出来的标签。请你结合工作描述信息和模型预测标签之间的契合度打个分，分值要求: 1~5分，1分为所有标签不相关，5分为所有标签都相关。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n",
        6: "现在我要提供你一份模型预测标签，这份标签是根据项目描述和项目职责信息，由模型标注出来的标签。请你结合项目描述和项目职责信息和模型预测标签之间的契合度打个分，分值要求: 1~5分，1分为所有标签不相关，5分为所有标签都相关。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n",
        7: "现在我要提供你一个模型行业大类预测结果，这个预测是根据岗位名称、岗位职责和任职要求信息，从可选项市场营销、产品/项目/运营、通信/硬件、咨询/管理、人力/财务/行政、供应链/物流、机械/制造、视觉/交互/设计、金融、软件开发、教育/科研、生物医药，判断得到的最相关行业大类。请你结合岗位名称、岗位职责和任职要求信息和模型预测的行业大类之间的契合度打个分，分值要求: 1~5分，1分为最不相关，5分为最相关。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n",
        8: "现在我要提供你一个模型行业大类预测结果，这个预测是根据工作描述信息，从可选项市场营销、产品/项目/运营、通信/硬件、咨询/管理、人力/财务/行政、供应链/物流、机械/制造、视觉/交互/设计、金融、软件开发、教育/科研、生物医药，判断得到的最相关行业大类。请你结合工作描述信息和模型预测标签之间的契合度打个分，分值要求: 1~5分，1分为最不相关，5分为最相关。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n",
        9: "现在我要提供你一个模型行业大类预测结果，这个预测是根据项目描述和项目职责信息，从可选项市场营销、产品/项目/运营、通信/硬件、咨询/管理、人力/财务/行政、供应链/物流、机械/制造、视觉/交互/设计、金融、软件开发、教育/科研、生物医药，判断得到的最相关行业大类。请你结合项目描述和项目职责信息和模型预测标签之间的契合度打个分，分值要求: 1~5分，1分为最不相关，5分为最相关。输出格式: 请在开头给出你的分数，形式为 xx分。紧接着给出你打分的理由。\n\n"
    }
    return prompts.get(prompt_type, "")

# def construct_prompt(content, prompt_type):
#     if prompt_type == 1:
#         true_label = " ".join(content["真值标签"])
#         predict_label = content["模型预测"]
#         return "### 真实标签: {}\n\n### 模型预测标签: {}\n\n".format(true_label, predict_label)
#     elif prompt_type == 2:
#         true_label = " ".join(content["真值标签"])
#         predict_label = content["模型预测"]
#         responsibility = content["岗位职责"]
#         requirment = content["任职要求"]
#         return "### 岗位职责: {}\n\n### 任职要求: {}\n\n### 真实标签: {}\n\n### 模型预测标签: {}\n\n".format(responsibility, requirment, true_label, predict_label)
#     elif prompt_type == 3:
#         predict_label = content["模型预测"]
#         responsibility = content["岗位职责"]
#         requirment = content["任职要求"]
#         return "### 岗位职责: {}\n\n### 任职要求: {}\n\n### 模型预测标签: {}\n\n".format(responsibility, requirment, predict_label)
#     elif prompt_type == 4:
#         predict_label = content["模型预测"]
#         description = content["工作描述"]
#         return "### 工作描述: {}\n\n### 模型预测标签: {}\n\n".format(description, predict_label)
#     elif prompt_type == 5:
#         true_label = " ".join(content["真值标签"])
#         predict_label = content["模型预测"]
#         description = content["工作描述"]
#         return "### 工作描述: {}\n\n### 真实标签: {}\n\n### 模型预测标签: {}\n\n".format(description, true_label, predict_label)
#     else:
#         return ""

def construct_prompt(content, prompt_type):
    if prompt_type >= 7:
        # return content["prompt"] + "### 模型预测标签：" + content["模型预测"] + "\n\n"
        strings = ",".join(content["模型预测"])
        return content["prompt"] + "### 模型预测标签：" + strings + "\n\n"
    
    elif prompt_type >= 4 and prompt_type <= 6:
        strings = ",".join(content["模型预测"])
        return content["prompt"] + "### 模型预测标签：" + strings + "\n\n"


def gpt_score(contents):
    try_again = []
    scores = []
    replys = []
    for i,content in enumerate(tqdm.tqdm(contents)):
        chat = ""
        reply=None
        prompt = construct_prompt(content,args.prompt_type)
        #print(prompt)
        try:
            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": get_system_prompt(args.prompt_type)},
                    {"role": "user", "content": prompt}],
                model=MODLE_TYPE
                )
            reply = chat.choices[0].message.content.strip()
            replys.append(reply)
            print(reply)

        except Exception as e:
            print("An error occurred:", str(e))
            print("something wrong with reply")
        
        if reply is None:
            print(f"The {i}th data cannot be scored normally")
            try_again.append(content)
        else:
            # Regular expression pattern
            pattern = r"(\d+)分"
            # Search Match
            match = re.search(pattern, reply)
            if match:
                score = match.group(1)
                print(f"The extracted score is: {score}")
            else:
                score = 0
                print("No matching scores found.")
            scores.append(int(score))
    
    print(f"Number of successful scoring：{len(scores)}")
    return scores,replys


def main(args):
    with open(args.input_file,"r",encoding='utf-8') as f:
        datas = json.load(f)

    if args.debug:
        datas = datas[:4]

    #print(get_system_prompt(args.prompt_type))
    final_score,replys = gpt_score(datas)
    avg_score = sum(final_score) / len(final_score)
    result = {"avg_score":avg_score,
              "scores":final_score,
              "responses":replys}
    with open(args.output_file,'w+',encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        default="/mnt/nvme1/lzp/tmp/deepseek-llm-7b-chat_jd.json")
    parser.add_argument("--output_file",
                        type=str,
                        default=None)
    parser.add_argument("--prompt_type",
                        type=int,
                        default=3)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)


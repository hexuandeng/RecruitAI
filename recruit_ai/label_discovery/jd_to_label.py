import json
import argparse
from tqdm import tqdm
from utils import LabelInference, JOBCATEGORY

LABEL_PER_ASK = 2
ROOT = ""

def construct_user_prompt(data):
    if data["岗位职责"].strip() == data["任职要求"].strip():
        prompt = "### 工作岗位：{}\n\n### 岗位职责：{}\n\n".format(
            data["需求岗位"], data["岗位职责"].strip())
    else:
        prompt = "### 工作岗位：{}\n\n### 岗位职责：{}\n\n### 任职要求：{}\n\n".format(
            data["需求岗位"], data["岗位职责"].strip(), data["任职要求"].strip())
    return prompt


def main(args):
    task = LabelInference(args, [0, 1, 2, 3, 4, 5, 6, 7], gpu_per_process=1, int4=True)

    with open(args.input_file, "r", encoding="utf-8") as f:
        job_name_datas = json.load(f)
    if args.debug:
        args.output_file = f'{ROOT}/dhx/labeling/code/debug'
        job_name_datas = job_name_datas[: 50]

    cnt = 0
    for data in tqdm(job_name_datas):
        task.push(construct_user_prompt(data), JOBCATEGORY, 1, cnt)
        cnt += 1

    print("Start Processing!")
    results = []
    trains = []
    cnt = 0
    for data in job_name_datas:
        trains.append({
            "prompt_id": 1,
            "input_string": construct_user_prompt(data).strip(),
            "labels": JOBCATEGORY,
            "answer": task.get_by_index(cnt)
        })
        data["所属大类"] = trains[-1]["answer"]
        results.append(data)
        cnt += 1

    with open(args.output_file + "1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    with open(args.output_file + "_group.json", "w", encoding="utf-8") as f:
        json.dump(trains, f, ensure_ascii=False, indent=4)

    # task.wait_finish()

    # task = LabelInference(args, [0, 1, 2, 3, 4, 5, 6, 7])

    with open(args.output_file + "1.json", "r", encoding="utf-8") as f:
        job_name_datas = json.load(f)
    if args.debug:
        job_name_datas = job_name_datas[: 50]

    with open(args.label_file, 'r', encoding='utf-8') as file:
        labels_map = json.load(file)

    cnt = 0
    for data in tqdm(job_name_datas):
        labels = []
        for it in data["所属大类"]:
            labels += labels_map[it[0]]
        labels = list(set(labels))
        for it in range(int((len(labels) - 1) / LABEL_PER_ASK) + 1):
            cnt_labels = labels[it * LABEL_PER_ASK: it * LABEL_PER_ASK + LABEL_PER_ASK]
            task.push(construct_user_prompt(data), cnt_labels + ["NOT FOUND"], 2, cnt)
            cnt += 1

    cnt = 0
    results = []
    trains = []
    for data in job_name_datas:
        cnt_results = []
        labels = []
        for it in data["所属大类"]:
            labels += labels_map[it[0]]
        labels = list(set(labels))
        for it in range(int(len(labels) / LABEL_PER_ASK)):
            cnt_labels = labels[it * LABEL_PER_ASK:it * LABEL_PER_ASK + LABEL_PER_ASK]
            trains.append({
                "prompt_id": 2,
                "input_string": construct_user_prompt(data).strip(),
                "labels": cnt_labels + ["NOT FOUND"],
                "answer": task.get_by_index(cnt)
            })
            cnt_results += trains[-1]["answer"]
            cnt += 1
        data["技术标签"] = cnt_results
        results.append(data)

    with open(args.output_file + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    with open(args.output_file + "_label.json", "w", encoding="utf-8") as f:
        json.dump(trains, f, ensure_ascii=False, indent=4)

    task.wait_finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default=f'{ROOT}/LLMs/Qwen2-72B-Instruct')
    parser.add_argument(
        '--input_file',
        type=str,
        default=f'{ROOT}/data/JD库/需求数据.json')
    parser.add_argument(
        '--label_file',
        type=str,
        default=f'{ROOT}/data/标签库/big_small_label.json'
    )
    parser.add_argument('--output_file', type=str, default=f'{ROOT}/dhx/labeling/output/2_jd_to_label')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--max_words", type=int, default=2048)
    args = parser.parse_args()
    main(args)

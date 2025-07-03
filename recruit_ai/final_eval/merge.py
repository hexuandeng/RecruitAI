import argparse
import json


def main(args: argparse.Namespace):
    with open(args.baseline_path, 'r', encoding='utf-8') as f:
        baseline_datas = json.load(f)
    with open(args.our_path, 'r', encoding='utf-8') as f:
        our_datas = json.load(f)
    for baseline, our in zip(baseline_datas, our_datas):
        baseline["our"] = our["our"]
        baseline["our_response_time"] = our["our_response_time"]
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_datas, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", type=str)
    parser.add_argument("--our_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    main(args)

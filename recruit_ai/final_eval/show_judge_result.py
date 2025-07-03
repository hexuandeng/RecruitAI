import argparse
import json


def map_response_time_to_score(response_time):
    """
    Mapping response time to a score
    """
    if response_time <= 2:
        return 10
    elif response_time <= 5:
        return 8
    elif response_time <= 15:
        return 6
    elif response_time <= 30:
        return 4
    elif response_time <= 60:
        return 2
    else:
        return 0  # If the time exceeds 60 seconds, the score is 0 (unacceptable)


def tongji(data):
    win_count = ping_count = loss_count = 0
    for item in data:
        if item["match_win_rates"] == 1:
            win_count += 1
        elif item["match_win_rates"] == -1:
            loss_count += 1
        else:
            ping_count += 1
    print(f"Match win: {win_count}， {win_count/len(data)}")
    print(f"Match loss: {loss_count}， {loss_count/len(data)}")
    print(f"Match tie: {ping_count}， {ping_count/len(data)}")

    win_count = ping_count = loss_count = 0
    for item in data:
        if item["difficulty_win_rates"] == 1:
            win_count += 1
        elif item["difficulty_win_rates"] == -1:
            loss_count += 1
        else:
            ping_count += 1
    print(f"Difficulty fit win: {win_count}，  {win_count/len(data)}")
    print(f"Difficulty fit loss: {loss_count}， {loss_count/len(data)}")
    print(f"Difficulty fit tie: {ping_count}， {ping_count/len(data)}")

    win_count = ping_count = loss_count = 0
    for item in data:
        if item["c_r_win_rate"] == 1:
            win_count += 1
        elif item["c_r_win_rate"] == -1:
            loss_count += 1
        else:
            ping_count += 1
    print(f"Correctness and rationality win: {win_count}，  {win_count/len(data)}")
    print(f"Correctness and rationality loss: {loss_count}， {loss_count/len(data)}")
    print(f"Correctness and rationality tie: {ping_count}， {ping_count/len(data)}")

    win_count = ping_count = loss_count = 0
    for item in data:
        if item["our_response_time"] < item["base_response_time"]:
            win_count += 1
        elif item["our_response_time"] > item["base_response_time"]:
            loss_count += 1
        else:
            ping_count += 1
    print(f"Inference speed win: {win_count}，  {win_count/len(data)}")
    print(f"Inference speed loss: {loss_count}， {loss_count/len(data)}")
    print(f"Inference speed tie: {ping_count}， {ping_count/len(data)}")


def zong_tongji(data):
    win_count = ping_count = loss_count = 0
    for item in data:
        match = item["match_win_rates"]
        difficulty = item["difficulty_win_rates"]
        c_r = item["c_r_win_rate"]
        inference_cost = 1 if item["our_response_time"] < item[
            "base_response_time"] else -1
        duplication = item["duplication_win_rates"]
        our = item['practice_theory_score_list']['recruitai']
        baseline = item['practice_theory_score_list']['baseline']
        our_score = sum(our) / len(our) if len(our) > 0 else 0
        baseline_score = sum(baseline) / len(baseline) if len(
            baseline) > 0 else 0
        if our_score > baseline_score:
            practice = 1
        elif our_score < baseline_score:
            practice = -1
        else:
            practice = 0

        try:
            total = match + difficulty + c_r + inference_cost + duplication + practice
        except:
            pass

        if total > 0:
            win_count += 1
        elif total < 0:
            loss_count += 1
        else:
            ping_count += 1
    print(f"Overall win: {win_count}，  {win_count/len(data)}")
    print(f"Overall loss: {loss_count}， {loss_count/len(data)}")
    print(f"Overall ping: {ping_count}， {ping_count/len(data)}")


def print_result(metric: str, win: int, lose: int, tie: int):
    total = win + lose + tie
    print(f"{metric} win: {win}，  {win/total}")
    print(f"{metric} loss: {lose}， {lose /total}")
    print(f"{metric} tie: {tie}， {tie /total}")
    print("*" * 30)


def avg_tongji(datas):

    recruitai_data = [data['match_score']['recruitai'] for data in datas]
    recruitai_avg = sum(recruitai_data) / len(recruitai_data)
    baseline_data = [data['match_score']['baseline'] for data in datas]
    baseline_avg = sum(baseline_data) / len(baseline_data)
    print(f"Average match score of recruitai: {recruitai_avg}")
    print(f"Average match score of baseline: {baseline_avg}")
    print("*" * 30)
    recruitai_data = [data['difficulty_score']['recruitai'] for data in datas]
    recruitai_avg = sum(recruitai_data) / len(recruitai_data)
    baseline_data = [data['difficulty_score']['baseline'] for data in datas]
    baseline_avg = sum(baseline_data) / len(baseline_data)
    print(f"Average difficulty of recruitai: {recruitai_avg}")
    print(f"Average difficulty of baseline: {baseline_avg}")
    print("*" * 30)

    recruitai_data = [
        data['correct_rationality_total_score']['recruitai'] for data in datas
    ]
    recruitai_avg = sum(recruitai_data) / len(recruitai_data)
    baseline_data = [
        data['correct_rationality_total_score']['baseline'] for data in datas
    ]
    baseline_avg = sum(baseline_data) / len(baseline_data)
    print(f"Average value of correctness and rationality of recruitai: {recruitai_avg}")
    print(f"Average value of correctness and rationality of baseline: {baseline_avg}")
    print("*" * 30)
    recruitai_data = [data['duplication_score']['recruitai'] for data in datas]
    recruitai_avg = sum(recruitai_data) / len(recruitai_data)
    baseline_data = [data['duplication_score']['baseline'] for data in datas]
    baseline_avg = sum(baseline_data) / len(baseline_data)
    print(f"Average repeatability of recruitai: {recruitai_avg}")
    print(f"Average repeatability of baseline: {baseline_avg}")
    print("*" * 30)
    recruitai_data = [
        map_response_time_to_score(data['our_response_time']) for data in datas
    ]
    recruitai_avg = sum(recruitai_data) / len(recruitai_data)
    baseline_data = [
        map_response_time_to_score(data['base_response_time'])
        for data in datas
    ]
    baseline_avg = sum(baseline_data) / len(baseline_data)
    print(f"The average inference speed of recruitai: {recruitai_avg}")
    print(f"The average inference speed of baseline: {baseline_avg}")
    print("*" * 30)
    recruitai_data = [
        sum(data['practice_theory_score_list']['recruitai']) /
        len(data['practice_theory_score_list']['recruitai'])
        if len(data['practice_theory_score_list']['recruitai']) > 0 else 0
        for data in datas
    ]
    recruitai_avg = sum(recruitai_data) / len(recruitai_data)
    baseline_data = [
        sum(data['practice_theory_score_list']['baseline']) /
        len(data['practice_theory_score_list']['baseline'])
        if len(data['practice_theory_score_list']['baseline']) > 0 else 0
        for data in datas
    ]
    baseline_avg = sum(baseline_data) / len(baseline_data)
    print(f"Practical theory combines average of recruitai: {recruitai_avg}")
    print(f"Practical theory combines average of baseline: {baseline_avg}")


def main(args: argparse.Namespace):
    with open(args.input_file_exam, 'r', encoding='utf-8') as f:
        exam_datas = json.load(f)
    with open(args.input_file_interview, 'r', encoding='utf-8') as f:
        interview_datas = json.load(f)
    datas = exam_datas + interview_datas
    if 'match' in args.metrics:
        win = lose = tie = 0
        for data in datas:
            if abs(data['match_score']['recruitai'] -
                   data['match_score']['baseline']) <= args.tie:
                tie += 1
            elif data['match_score']['recruitai'] > data['match_score'][
                    'baseline']:
                win += 1
            else:
                lose += 1
        print_result('Matching degree', win, lose, tie)
    if 'difficulty' in args.metrics:
        win = lose = tie = 0
        for data in datas:
            if data['difficulty_score']['recruitai'] == data[
                    'difficulty_score']['baseline']:
                tie += 1
            elif data['difficulty_score']['recruitai'] > data[
                    'difficulty_score']['baseline']:
                win += 1
            else:
                lose += 1
        print_result('Difficulty fit', win, lose, tie)
    if 'c_r' in args.metrics:
        win = lose = tie = 0
        for data in datas:
            if abs(data['correct_rationality_total_score']['recruitai'] -
                   data['correct_rationality_total_score']['baseline']
                   ) <= args.tie:
                tie += 1
            elif data['correct_rationality_total_score']['recruitai'] > data[
                    'correct_rationality_total_score']['baseline']:
                win += 1
            else:
                lose += 1
        print_result('Correctness and rationality', win, lose, tie)
    if 'dup' in args.metrics:
        win = lose = tie = 0
        for data in datas:
            if abs(data['duplication_score']['recruitai'] -
                   data['duplication_score']['baseline']) <= args.tie:
                tie += 1
            elif data['duplication_score']['recruitai'] > data[
                    'duplication_score']['baseline']:
                win += 1
            else:
                lose += 1
        print_result('Repeatability', win, lose, tie)
    if 'p_t' in args.metrics:
        win = lose = tie = 0
        for data in datas:
            our = data['practice_theory_score_list']['recruitai']
            baseline = data['practice_theory_score_list']['baseline']
            our_score = sum(our) / len(our) if len(our) > 0 else 0
            baseline_score = sum(baseline) / len(baseline) if len(
                baseline) > 0 else 0
            if abs(our_score - baseline_score) <= args.tie:
                tie += 1
            elif our_score > baseline_score:
                win += 1
            else:
                lose += 1
        print_result('Combining theory with practice', win, lose, tie)
    if 'inf' in args.metrics:
        win = lose = tie = 0
        for data in datas:
            if data['our_response_time'] == data['base_response_time']:
                tie += 1
            elif data['our_response_time'] < data['base_response_time']:
                win += 1
            else:
                lose += 1
        print_result('Inference speed', win, lose, tie)

    print("************************")
    print("Average Statistics")
    avg_tongji(datas)

    print("************************")
    print("Overall Statistics")
    zong_tongji(datas)
    print("************************")
    print("Overall statistics - remove None")
    for item in datas:
        if item["our"] is None or item["baseline"] is None:
            datas.remove(item)
    zong_tongji(datas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_exam', type=str)
    parser.add_argument('--input_file_interview', type=str)
    parser.add_argument('--metrics', type=str)
    parser.add_argument('--tie', type=float, default=0)
    main(parser.parse_args())

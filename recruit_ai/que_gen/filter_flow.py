import argparse
import logging
import multiprocessing
from functools import partial
from recruit_ai.que_gen.utils import construct_question, filter_comb, load_data_mode, save_data_mode
from tqdm import tqdm

input_path1 = "datas/question_set_deepseek_0410.json"
input_path2 = "questions_filter/data/test.json"
test_path = "recruit_ai/que_gen/test_data/output_img.json"
output_path1 = "recruit_ai/que_gen/test_result/filtered.json"
output_path2 = "recruit_ai/que_gen/test_result/need.json"
output_path3 = "recruit_ai/que_gen/test_result/exception.json"

data_base_path = '../datas/recruit_ai.db'

def process_question(question, args):
    """
    Functions that handle a single problem, for multi-process processing
    """
    try:
        mode = list(map(int, args.mode_choice.split(',')))
        results = {
            "choice": [],
            "fill_bank": [],
            "judge": []
        }
        results_id = {
            "choice": [],
            "fill_bank": [],
            "judge": []
        }
        exception = {
            "choice": [],
            "fill_bank": [],
            "judge": []
        }

        if "choice" in question["types"] and question["types"]["choice"] != None:
            result = filter_comb(question["types"]["choice"],0,
                                        mode,
                                        model = args.model,
                                        frequency_penalty=0,
                                        temperature=0)
            if result:
                exception["choice"].append(construct_question(question["types"]["choice"],0))
            else:
                results_id["choice"].append(question["id"])
                results["choice"].append(construct_question(question["types"]["choice"],0))
                    
        if  "fill_bank" in question["types"] and question["types"]["fill_bank"] != None:
            result = filter_comb(question["types"]["fill_bank"],1,
                                        mode,
                                        model = args.model,
                                        frequency_penalty=0,
                                        temperature=0)
            if result:
                exception["fill_bank"].append(construct_question(question["types"]["fill_bank"],1))
            else:
                results_id["fill_bank"].append(question["id"])
                results["fill_bank"].append(construct_question(question["types"]["fill_bank"],1))
                    
        if  "judge" in question["types"] and question["types"]["judge"] != None:
            result = filter_comb(question["types"]["judge"].copy(), 2,
                                mode,
                                model=args.model,
                                frequency_penalty=0,
                                temperature=0)
            i = 0
            if False in result:
                result = [False for _ in result]
            for c in result:
                if c:
                    exception["judge"].append((question["types"]["judge"]["question"][i], question["types"]["judge"]["answer"][i]))
                else:
                    results["judge"].append((question["types"]["judge"]["question"][i], question["types"]["judge"]["answer"][i]))
                    results_id["judge"].append((question["id"], i))
                i += 1
        return results, results_id, exception
    except Exception as e:
        logging.error(f"{e}")
        return None

def main(parser):
    args = parser.parse_args()
    questions = load_data_mode(args.input_path)
    #questions = load_data_mode(input_path2)
    
    if args.debug:
        questions = questions[:100]
        
    if "gpt" in args.model:
        processes_num = 30
    else:
        processes_num = 30
        
    process_func = partial(process_question, args=args)
    print(multiprocessing.cpu_count())
    with multiprocessing.Pool(processes = processes_num) as pool:
        processed_results = list(tqdm(pool.imap(process_func, questions), total=len(questions)))
    
    pool.close()
    pool.join()

    results = {
        "choice": [],
        "fill_bank": [],
        "judge": []
    }
    results_id = {
        "choice": [],
        "fill_bank": [],
        "judge": []
    }
    exception = {
        "choice": [],
        "fill_bank": [],
        "judge": []
    }

    for result, result_id, exc in processed_results:
        results["choice"].extend(result["choice"])
        results["fill_bank"].extend(result["fill_bank"])
        results["judge"].extend(result["judge"])

        results_id["choice"].extend(result_id["choice"])
        results_id["fill_bank"].extend(result_id["fill_bank"])
        results_id["judge"].extend(result_id["judge"])

        exception["choice"].extend(exc["choice"])
        exception["fill_bank"].extend(exc["fill_bank"])
        exception["judge"].extend(exc["judge"])

    if not args.debug:
        save_data_mode(results, output_path1)
        save_data_mode(results_id, output_path2)
        save_data_mode(exception, args.output_path)
    else:
        save_data_mode(results, output_path1 + "debug")
        save_data_mode(results_id, output_path2 + "debug")
        save_data_mode(exception, args.output_path + "debug")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=" ")
    
    parser.add_argument('--model', 
                        type=str, 
                        default="Qwen2.5-72B-Instruct",
                        help="model name")

    parser.add_argument('--input_path', 
                        type=str, 
                        default="recruit_ai/que_gen/test_data/output_50n.json",
                        help="input_path")

    parser.add_argument('--output_path', 
                        type=str, 
                        default="recruit_ai/que_gen/test_result/exception.json",
                        help="output_path")
    
    parser.add_argument('--mode_choice', 
                        type=str, 
                        default="1,2,3,4,5",
                        help="Filter selection, separated by \",\", 1-5 represent image filtering, answer correctness filtering, answer existence filtering, question completeness filtering, question appropriateness filtering")
    
    parser.add_argument('--debug', 
                        action='store_true', 
                        help="debug")
    main(parser)

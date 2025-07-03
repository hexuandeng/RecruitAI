import argparse
import json
import os
import threading
from collections import defaultdict
from functools import *
from time import time
from typing import Any, Dict, List, Tuple

import orjson
import requests
from datasets import Dataset, load_dataset
from flask import Flask, jsonify, request

app = Flask(__name__)

PATHS = {
    'jd': '../data/jd.json',
    'work': '../data/fb_talent_database_work.json',
    'project': '../data/fb_talent_database_project.json',
    'written_questions': '../data/generated_written_questions.json',
    'label_map': '../data/big_small_label.json',
}


def load_data_by_id(
    data_path: str,
    key: str,
    id: int | str,
) -> List[Dict[str, int | str]]:
    """
    Loads data from the JSON file at the specified path and filters out matching data items based on the given key and ID.

    parameter:
    data_path (str): The path to the JSON file to load.
    key (str): Keys used for matching.
    id (int): ID to match.

    return:
    List[Dict[str, int | str]]: A list containing all data items with matching IDs.
    """
    try:
        response = requests.post(f'http://127.0.0.1:16456/load_json',
                                 json={
                                     "data_path": data_path,
                                     "key": key,
                                     "id": id
                                 })
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        print(e)
        raise Exception(f"Request to classifier failed: {e}") from e
    except json.JSONDecodeError as e:
        print(e)
        raise Exception(f"Failed to decode JSON response: {e}") from e


class DataManager:
    def __init__(self):
        self.datas = {}
        self.cache = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        self.cache_lock = threading.Lock()
        for data_path in PATHS.values():
            if data_path == PATHS["written_questions"]:
                continue
            print(f'loading dataset: {data_path}')
            start_time = time()
            try:
                self.datas[data_path] = load_dataset(
                    "json", data_files=data_path)["train"]
            except:
                print(data_path)
                with open(data_path, 'rb') as f:  # orjson requires reading in binary mode
                    self.datas[data_path] = orjson.loads(
                        f.read())  # Loading the entire JSON file at once
            end_time = time()
            print(f'loading time: {end_time - start_time:.5f} seconds')

        self.cache_data_by_key(PATHS["jd"], "需求编号")
        self.cache_data_by_key(PATHS["work"], "pid")
        self.cache_data_by_key(PATHS["project"], "pid")
        print("Cached File & Key")
        for k, v in self.cache.items():
            for k0, v0 in v.items():
                print(k, k0)

    def map_by_key(self, data_list, idx):
        local_cache = {}
        for cnt, key_value in zip(idx, data_list):
            if key_value not in local_cache:
                local_cache[key_value] = []
            local_cache[key_value].append(cnt)
        return {
            "value": list(local_cache.keys()),
            "match": list(local_cache.values())
        }

    def cache_data_by_key(
            self,
            data_path: str,
            cache_key: str,
            n: int = 4,  # Number of threads
    ) -> List[Dict[str, int | str]]:
        print(f'Caching {data_path} {cache_key}!')
        start_time = time()
        if isinstance(self.datas[data_path], Dataset):
            dataset = self.datas[data_path]
            # print(self.datas[data_path])
            new_dataset = dataset.map(self.map_by_key,
                                      input_columns=cache_key,
                                      with_indices=True,
                                      batched=True,
                                      batch_size=500,
                                      num_proc=n,
                                      remove_columns=dataset.column_names)
            print(new_dataset)
            print(new_dataset[:3])
            for it in new_dataset:
                key_value, cnts = it["value"], it["match"]
                self.cache[data_path][cache_key][key_value].extend(cnts)
            end_time = time()
            print(
                f'Caching {data_path} {cache_key} time: {end_time - start_time:.5f} seconds\n\n'
            )
        else:
            data_list = self.datas[data_path]
            data_len = len(data_list)
            chunk_size = (data_len + n -
                          1) // n  # Ceiling division to determine chunk size

            # List to hold local caches from each thread
            local_caches = [None] * n

            def worker(index, start, end):
                local_cache = {}
                for cnt in range(start, end):
                    data = data_list[cnt]
                    key_value = data[cache_key]
                    if key_value not in local_cache:
                        local_cache[key_value] = []
                    local_cache[key_value].append(cnt)
                local_caches[index] = local_cache

            threads = []
            for i in range(n):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, data_len)
                t = threading.Thread(target=worker, args=(i, start, end))
                threads.append(t)
                t.start()

            # Wait for all threads to finish
            for t in threads:
                t.join()

            # Merge local caches into the shared self.cache
            for local_cache in local_caches:
                for key_value, cnts in local_cache.items():
                    with self.cache_lock:
                        self.cache[data_path][cache_key][key_value].extend(
                            cnts)

            end_time = time()
            print(
                f'Caching {data_path} {cache_key} time: {end_time - start_time:.5f} seconds\n\n'
            )

    def load_data_by_id(
        self,
        data_path: str,
        key: str,
        id: int | str,
    ) -> List[Dict[str, int | str]]:
        """
        Loads data from the JSON file at the specified path and filters out matching data items based on the given key and ID.

        parameter:
        data_path (str): The path to the JSON file to load.
        key (str): The key to use for matching.
        id (int): The ID to match.

        return:
        List[Dict[str, int | str]]: A list containing all data items with matching IDs.
        """
        if data_path in self.cache and key in self.cache[data_path]:
            # end_time = time()
            # print(f'matching id time: {end_time - start_time:.5f} seconds\n\n')
            return [
                self.datas[data_path][i]
                for i in self.cache[data_path][key][id]
            ]

        start_time = time()
        if isinstance(self.datas[data_path], Dataset):
            result = self.datas[data_path].filter(lambda x: x[key] == id,
                                                  num_proc=32).to_list()
        else:
            result = []
            for data in self.datas[data_path]:
                if data[key] == id:
                    result.append(data)
        end_time = time()
        print(f'matching id time: {end_time - start_time:.5f} seconds\n\n')
        return result

    def save_generated_written_questions(self, data: Dict[str, Any]):
        print(data)
        try:
            self.datas[PATHS['written_questions']].add_item(data)
            with open(PATHS['written_questions'], 'w', encoding='utf-8') as f:
                json.dump(self,
                          self.datas[PATHS['written_questions']],
                          f,
                          ensure_ascii=False,
                          indent=4)
            return 0
        except:
            return -1


@app.route('/load_json', methods=['POST'])
def load_json():
    json_data = request.get_json()
    try:
        print(f'request data: {json_data}')
        response = data_manager.load_data_by_id(**json_data)
        json_data = {'response': response}
        # print(f'response data: {json_data}')
        return jsonify(json_data), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/save_generated_written_questions', methods=['POST'])
def save_generated_written_questions() -> Tuple[Dict[str, int], int]:
    json_data = request.get_json()
    try:
        print(f'request data: {json_data}')
        response = data_manager.save_generated_written_questions(**json_data)
        json_data = {'code': response}
        print(f'response data: {json_data}')
        return jsonify(json_data), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


def main(args: argparse.Namespace):
    print('initializing load json...')
    global data_manager, port
    data_manager = DataManager()
    port = args.port
    print(f'Start server on port {port}')
    app.run(port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=16456)
    main(parser.parse_args())

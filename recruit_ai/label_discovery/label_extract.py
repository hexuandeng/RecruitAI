import json

file_path = "random_jd.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

print(len(data))
num = 0
labels = []
for d in data:
    l = d['answer'].split(',')
    for label in l:
        if label != "无":
            labels.append(label)
    num += 1
    if num >= 200:
        break

with open('jd_list.json', 'w', encoding='utf-8') as file:
    json.dump(labels, file, ensure_ascii=False, indent=4)

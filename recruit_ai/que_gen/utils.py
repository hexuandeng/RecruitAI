import json
import re
from typing import Dict, List

from transformers import AutoTokenizer
from recruit_ai.data_utils.utils import eval_choice_or_judge_questions, eval_fill_in_questions
from recruit_ai.data_utils.request_model import chat

PROMPTS = [
    #选择
    [
    "你是一名专业的答题人，你的职责是针对提供的选择题题目，给出题目答案，若多选，选项之间不要使用任何分隔符，只需输出选项即可，无需其他输出。",
    "你是一名专业的审题人，你的职责是针对提供的选择题题目及答案，判断题目是否一定需要额外添加图片或表格才能回答，请先进行详细的分析，最后输出是或否。",
    "你是一名专业的审题人，你的职责是针对提供的选择题题目，判断题目中是否直接包含了正确选项的答案，请先进行详细的分析，最后输出是或否。",
    """
    请评估以下选择题是否符合质量要求。严格按照格式输出，不需要分角度逐项分析，只需综合评估后给出一个简短总结，并判断题目是否质量合格。

    评估角度（供参考，仅用于综合评估，不要逐项分析）：
    1. **答案正确性**：答案是否正确，并能被客观验证。
    2. **考察意义**：题目是否具备知识点考察的价值，而不是简单的记忆性问题或无意义的挖空（例如过于直白、常识性或与题目无关的填空）。
    3. **语言表达清晰度**：题目的语言是否清楚、简洁，无语法错误。
    4. **难度适宜性**：题目难度是否适中，既不过于简单也不超出目标受众的理解范围。

    ### 严格输出格式：
    总结: <简短总结>
    是否质量合格: 是/否
    输入选择题如下：
    """,
    "",
    "现在我将给你提供一个技术标签，请你根据我提供的一道题目及其正确答案来判断该技术标签是否属于这道题目的考察范围。可选项为：是 否。请注意你只需要回答是/否即可。",
    "我将给你提供一道选择题题目及其正确答案，请你从下列技术标签中选择一项与之最为契合的选项。"
    ],
    #填空
    [
    "你是一名专业的答题人，你的职责是针对提供的填空题题目，给出题目答案，只需输出答案即可，无需其他输出。",
    "你是一名专业的审题人，你的职责是针对提供的填空题题目及答案，判断题目是否一定需要额外添加图片或表格才能回答，请先进行详细的分析，最后输出是或否。",
    "你是一名专业的审题人，你的职责是针对提供的填空题题目，判断题目中是否直接包含了正确答案，请先进行详细的分析，最后输出是或否。",
    """
    请评估以下填空题是否符合质量要求。严格按照格式输出，不需要分角度逐项分析，只需综合评估后给出一个简短总结，并判断题目是否质量合格。

    评估角度（供参考，仅用于综合评估，不要逐项分析）：
    1. **答案正确性**：答案是否正确，并能被客观验证。
    2. **考察意义**：题目是否具备知识点考察的价值，而不是简单的记忆性问题或无意义的挖空（例如过于直白、常识性或与题目无关的填空）。
    3. **语言表达清晰度**：题目的语言是否清楚、简洁，无语法错误。
    4. **难度适宜性**：题目难度是否适中，既不过于简单也不超出目标受众的理解范围。

    ### 严格输出格式：
    总结: <简短总结>
    是否质量合格: 是/否
    输入填空题如下：
    """,
    "你是一名专业的审题人，你的职责是针对提供的选择题及由其改编的填空题，判断改编题目是否包含了原题目所有解题信息，只需回答是否即可，无需其他输出。",
    "你是一名专业的HR。现在我将给你提供一个技术标签，请你根据我提供的一道题目及其正确答案来判断该技术标签是否属于这道题目的考察范围。可选项为：是 否。请注意你只需要回答是/否即可。",
    "我将给你提供一道填空题题目及其正确答案，请你从下列技术标签中选择一项与之最为契合的选项。"
    ],
    #判断
    [
    "你是一名专业的答题人，你的职责是针对提供的判断题题目，给出题目答案，只需输出是或否即可，无需其他输出。",
    "你是一名专业的审题人，你的职责是针对提供的判断题题目及答案，判断题目是否一定需要额外添加图片或表格才能回答，请先进行详细的分析，最后输出是或否。",
    "你是一名专业的审题人，你的职责是针对提供的判断题题目，判断题目中是否直接包含了正误判断答案，请先进行详细的分析，最后输出是或否。",
    """
    请评估以下判断题是否符合质量要求。严格按照格式输出，不需要分角度逐项分析，只需综合评估后给出一个简短总结，并判断题目是否质量合格。

    评估角度（供参考，仅用于综合评估，不要逐项分析）：
    1. **答案正确性**：答案是否正确，并能被客观验证。
    2. **考察意义**：题目是否具备知识点考察的价值，而不是简单的记忆性问题或无意义的挖空（例如过于直白、常识性或与题目无关的填空）。
    3. **语言表达清晰度**：题目的语言是否清楚、简洁，无语法错误。
    4. **难度适宜性**：题目难度是否适中，既不过于简单也不超出目标受众的理解范围。

    ### 严格输出格式：
    总结: <简短总结>
    是否质量合格: 是/否
    输入判断题如下：
    """,
    "你是一名专业的审题人，你的职责是针对提供的选择题及由其改编的判断题，判断改编题目是否包含了原题目所有解题信息，只需回答是否即可，无需其他输出。",
    "你是一名专业的HR。现在我将给你提供一个技术标签，请你根据我提供的一道题目及其正确答案来判断该技术标签是否属于这道题目的考察范围。可选项为：是 否。请注意你只需要回答是/否即可。",
    "我将给你提供一道判断题题目及其正确答案，请你从下列技术标签中选择一项与之最为契合的选项。"
    ],
    [
    "你是一名专业的答题人，你的职责是针对提供的简答题题目，给出题目答案，只需输出答案即可，无需其他输出。",
    "你是一名专业的审题人，你的职责是针对提供的题目及答案，判断题目是否一定需要额外添加图片或表格才能回答，只需回答是否即可，无需其他输出。",
    "你是一名专业的审题人，你的职责是针对提供的题目，判断题目中是否直接包含了答案，导致题目不够合理，请先进行分析，最后输出是或否。",
    "你是一名专业的审题人，你的职责是针对提供的题目及答案，判断题目是否完整且适合作为面试题，请先进行分析，最后输出是或否。",
    "你是一名专业的审题人，你的职责是针对提供的判断题题目及答案，判断题目是否信息足够完整来判断正误，请先进行分析，最后输出是或否。",
    "你是一名专业的HR。现在我将给你提供一个技术标签，请你根据我提供的一道题目及其正确答案来判断该技术标签是否属于这道题目的考察范围。可选项为：是 否。请注意你只需要回答是/否即可。"
    ]
]
SUSPS = [
    "这段",
    "下列",
    "上述",
    "描述",
    "接下来",
    "以下",
    "观点"
]

#Questions and Answers
def construct_question(question: Dict,
                       question_type: int) -> str|List[str]:
    #choice
    if question["question"] == None or question["answer"] == None :
        return ""
    if question_type == 0:
        question_text = "选择题题目:" + question["question"]+'\n'
        for i,option in enumerate(question["option"]):
            question_text = question_text+chr(ord('A')+i)+' '+option+'\n'
        question_text = question_text + "选择题答案:" + question["answer"]+'\n'
    #fill_bank
    elif question_type == 1:
        question_text = "填空题题目:" + question["question"]+'\n'
        #print(question_text)
        question_text = question_text + "填空题答案:" + question["answer"]+'\n'
        #print(question_text)
    #judge
    elif question_type == 2:
        question_text = []
        for _question , answer in zip(question["question"],question["answer"]):
            question_text_part = "判断题题目:" + _question+'\n'
            question_text_part = question_text_part+"判断题答案:" + answer+'\n'
            question_text.append(question_text_part)
    #single judge
    elif question_type == 4:
        question_text = "判断题题目:" + question["question"] +'\n'
        question_text = question_text+"判断题答案:" + question["answer"] +'\n'
    else:
        question_text = question["question"]+'\n'
        question_text = question_text + question["answer"]+'\n'
    return question_text

#Questions only
def construct_only_question(question: Dict,
                       question_type: int) -> str|List[str]:
    #choice
    if question_type == 0:
        question_text = "填空题题目:" + question["question"]+'\n'
        for i,option in enumerate(question["option"]):
            question_text = question_text+chr(ord('A')+i)+' '+option+'\n'
    #fill_bank
    elif question_type == 1:
        question_text = "填空题题目:" + question["question"]+'\n'
    #judge
    elif question_type == 2:
        question_text = []
        for _question in question["question"]:
            question_text_part = "判断题题目:" + _question+'\n'
            question_text.append(question_text_part)
    else:
        question_text = question["question"]+'\n'
    return question_text

#Determine the number of judgment questions that need to be filtered and filter them out
def confirm_judge_to_be_filtered(T_or_F: List[str],
                                 F: str|int) -> List:

    return [item for item,s in enumerate(T_or_F) if s != F] if isinstance(F,str) else [item for item,s in enumerate(T_or_F) if s >= F]

def construct_messages(system_prompt: str,
                       user_prompt: str) -> List[Dict[str,str]]:
    return  [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
    
def load_data_mode(data_file):
    data = json.load(open(data_file, "r",encoding='utf-8'))
    return data

def save_data_mode(data,data_file):
    with open(data_file,"w",encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def match_last_shi_or_fou(text: str) -> str:
    # Regular expression that matches the last "yes" or "no"
    
    matches = re.findall(r'(是|否)', text)
    
    # Returns the last match, or None if no match is found.
    return matches[-1] if matches else None

def find_first_number(text):
    match = re.search(r'(?<!\d)(10|[0-9])(?!\d)', text)  # Make sure to match a complete number from 1-10
    return int(match.group()) if match else 1

def filter_template(question: Dict[str,any],
    question_type: int,
    model: str,
    prompt: str,
    **kwargs) -> str|List[str]:
    if question_type != 2:
        
        messages = construct_messages(prompt,
                                      construct_question(question,question_type))
        
        completion = chat(messages= messages,
                        model = model,
                        **kwargs)
        print(construct_question(question,question_type))
        print(completion)
        return True if match_last_shi_or_fou(completion) == "是" else False
    else :
        question_text = construct_question(question,question_type)
        results = []
        for question_text_part in question_text:
            messages = construct_messages(prompt,
                                          question_text_part)
            
            results.append(True if match_last_shi_or_fou(chat(messages= messages,
                    model = model,
                    **kwargs)) == "是" else False)
            
        return results

#Filter the images are needed
def filter_img(question: Dict[str,any],
    question_type: int,
    model: str,
    **kwargs) -> str|List[str]:
    

    if question_type != 2:
        
        messages = construct_messages(PROMPTS[question_type][1],
                                      construct_question(question,question_type))
        
        completion = chat(messages= messages,
                        model = model,
                        **kwargs)
        print(construct_question(question,question_type))
        print(completion)
        return match_last_shi_or_fou(completion)
    else :
        question_text = construct_question(question,question_type)
        results = []
        for question_text_part in question_text:
            messages = construct_messages(PROMPTS[question_type][1],
                                          question_text_part)
            
            results.append(match_last_shi_or_fou(chat(messages= messages,
                    model = model,
                    **kwargs)))
            
        return results

#Make sure the answer is correct
def filter_wrong_answer(question:Dict[str,any],
    question_type: int,
    model: str,
    **kwargs) -> str|List[str]:

    if question_type != 2 :
        
        messages = construct_messages(PROMPTS[question_type][0],
                                      construct_only_question(question,question_type))
        
        # Completion API
        completion = chat(messages= messages,
                          model = model,
                          **kwargs)
        
        if question_type == 0:
            score = eval_choice_or_judge_questions([question], [completion])
        else:
            ai_answer = completion
            tokenizer = AutoTokenizer.from_pretrained('Qwen2-7B-Instruct')

            max_token_length = 3072

            encoded_text = tokenizer.encode(ai_answer, add_special_tokens=False)
            if len(encoded_text) > max_token_length:
                ai_answer = ""

            score = eval_fill_in_questions([question], [ai_answer], "Qwen2.5-72B-Instruct")
        return score[0],completion
    
    else:
        question_text = construct_only_question(question,question_type)
        results = []
        answers = []
        for i,question_text_part in enumerate(question_text):
            question_ = dict()
            question_["answer"] = question["answer"][i]
            messages = construct_messages(PROMPTS[question_type][0],
                                          question_text_part)

            # Completion API
            completion = chat(messages= messages,
                              model = model,
                              **kwargs)
            print(completion)
            completion = match_last_shi_or_fou(completion)
            score = eval_choice_or_judge_questions([question_], [completion])
            results.append(score[0])
            answers.append(completion)
            
        return results,answers

#Determine if the answer is in the question
def filter_answer_in_question(question: Dict[str,any],
    question_type: int,
    model: str,
    **kwargs) -> str|List[str]:

    if question_type != 2:
        if question_type == 1 and question["answer"] in question["question"]:
            return "否"
        messages = construct_messages(PROMPTS[question_type][2],
                                      construct_only_question(question,question_type))
        
        print(construct_only_question(question,question_type))
        completion = chat(messages= messages,
                          model = model,
                          **kwargs)
        print(completion)
        return match_last_shi_or_fou(completion)
    else :
        question_text = construct_only_question(question,question_type)
        results = []
        for question_text_part in question_text:
            messages = construct_messages(PROMPTS[question_type][2],
                                          question_text_part)
            print(question_text_part)
            completion = chat(messages= messages,
                              model = model,
                              **kwargs)
            print(completion)
            results.append(match_last_shi_or_fou(completion))
            
        return results

#Determine if content is missing
def filter_absence(question: Dict[str,any],
    question_type: int,
    model: str,
    **kwargs) -> str|List[str]:

    if question_type != 2:
        for susp in SUSPS:
            if susp in construct_only_question(question,question_type):

                messages = construct_messages(f"你是一名专业的审题人，你的职责是针对提供的题目，判断题目中“{susp}”所代指处是否存在上下文对应来保证题目完整性，请先进行详细的分析，最后输出是或否。",
                                              construct_only_question(question,question_type))
                
                completion = chat(messages= messages,
                                  model = model,
                                  **kwargs)
                print(construct_question(question,question_type))
                print(completion)
                return match_last_shi_or_fou(completion)
        return "是"
    else :
        question_text = construct_only_question(question,question_type)
        results = []
        for question_text_part in question_text:
            flag = 1
            for susp in SUSPS:
                if susp in question_text_part:

                    messages = construct_messages(f"你是一名专业的审题人，你的职责是针对提供的题目，判断题目中“{susp}”所代指处是否存在上下文对应来保证题目完整性，请先进行详细的分析，最后输出是或否。",
                                                  question_text_part)
                    
                    completion = chat(messages= messages,
                                      model = model,
                                      **kwargs)
                    print(construct_question(question,question_type))
                    print(completion)
                    results.append(match_last_shi_or_fou(completion))
                    flag = 0
                    break
            if flag:
                results.append("是")
            
        return results

#Determine if it is appropriate
def filter_proper(question: Dict[str,any],
    question_type: int,
    model: str,
    **kwargs) -> str|List[str]:

    if question_type != 2:
        messages = construct_messages(PROMPTS[question_type][3],
                                      construct_question(question,question_type))
        
        completion = chat(messages= messages,
                          model = model,
                          **kwargs)
        print(construct_question(question,question_type))
        print(completion)
        return match_last_shi_or_fou(completion)
    else :
        question_text = construct_only_question(question,question_type)
        results = []
        for question_text_part in question_text:
            messages = construct_messages(PROMPTS[question_type][3],
                                          question_text_part)
            completion = chat(messages= messages,
                                model = model,
                                **kwargs)
            print(construct_question(question,question_type))
            print(completion)
            results.append(match_last_shi_or_fou(completion))
            
        return results

def filter_compare(question_choice: Dict[str,any],
    question_another: Dict[str,any],
    question_type: int,
    model: str,
    **kwargs) -> str|List[str]:
    # Use multiple-choice questions and fill-in-the-blank questions/true-or-false questions to compare and determine whether the fill-in-the-blank questions/true-or-false questions are complete
    if question_type != 2:
        messages = construct_messages(PROMPTS[question_type][4],
                                      "###原选择题：" + construct_question(question_choice,0) + "\n\n\n" + 
                                      "###改编题目：" + construct_question(question_another,question_type) + "\n\n\n")
        
        completion = chat(messages= messages,
                          model = model,
                          **kwargs)
        print(construct_question(question_choice,0))
        print(construct_question(question_another,question_type))
        print(completion)
        return match_last_shi_or_fou(completion)
    else :
        question_text = construct_question(question_another,question_type)
        results = []
        for question_text_part in question_text:
            messages = construct_messages(PROMPTS[question_type][4],
                                      "###原选择题：" + construct_question(question_choice,0) + "\n\n\n" + 
                                      "###改编题目：" + question_text_part + "\n\n\n")
            completion = chat(messages= messages,
                                model = model,
                                **kwargs)
            print(construct_question(question_choice,0))
            print(construct_question(question_another,question_type))
            print(completion)
            results.append(match_last_shi_or_fou(completion))
            
        return results

def filter_block(question: Dict[str,any],
                 question_type: int) -> str|List[str]:
    if question_type != 2:
        text = question["question"]
        # Match and remove a pattern of increasing numbers line by line
        pattern = r"1\n2\n(?:\d+\n)*"
        match = re.search(pattern, text)
        # Use re.sub to replace the matched part with an empty string
        cleaned_text = re.sub(pattern, "", text)

        return match,cleaned_text
    else:
        result = []
        pattern = r"1\n2\n(?:\d+\n)*"
        match = False
        for text in question["question"]:
            cleaned_text = re.sub(pattern, "", text)
            match = re.search(pattern, text) or match
            result.append(cleaned_text)
        return match,result
    
def filter_comb(question: Dict[str,any],
    question_type: int,
    question_filter_choose: List[int],
    model: str,
    **kwargs) -> bool|List[bool]:
    
        if 'question_choice' in kwargs.keys():
            question_choice = kwargs["question_choice"]
            kwargs.pop('question_choice', None)
            
        if question_type == 2:
            result = [False for _ in range(len(question["question"]))]
            contins = [i for i in range(len(question["question"]))]
            
        if 1 in question_filter_choose:
            content1 = filter_img(question,
                                question_type,
                                model,
                                **kwargs)
            if content1 == None:
                return False

            if isinstance(content1, str):
                if content1 == "是":
                    return False
                elif all(element not in [2,3,4,5,6] for element in question_filter_choose):
                    return True
            else:
                if any(x is None for x in content1):
                    return result
                contin = confirm_judge_to_be_filtered(content1,"是")
                question["question"] = [question["question"][i] for i in contin]
                question["answer"] = [question["answer"][i] for i in contin]
                contins = [contins[i] for i in contin]
                
        if 2 in question_filter_choose:
            content2, answer = filter_wrong_answer(question,
                                                question_type,
                                                model,
                                                **kwargs)
            if content2 == None:
                return False

            if isinstance(content2, int):
                if content2 < 6:
                    return False
                elif all(element not in [3,4,5,6] for element in question_filter_choose):
                    return True
            else:
                if any(x is None for x in content2):
                    return result
                contin = confirm_judge_to_be_filtered(content2,6)
                question["question"] = [question["question"][i] for i in contin]
                question["answer"] = [question["answer"][i] for i in contin]
                contins = [contins[i] for i in contin]
                
        if 3 in question_filter_choose:
            print(3)
            content3 = filter_answer_in_question(question,
                                                    question_type,
                                                    model,
                                                    **kwargs) if question_type != 2 else ["否" for _ in question["question"]]
            print("content")
            print(content3)
            if content3 == None:
                return False

            if isinstance(content3, str):
                if content3 == "是":
                    return False
                elif all(element not in [4,5,6] for element in question_filter_choose):
                    return True
            else:
                if any(x is None for x in content3):
                    return result
                contin = confirm_judge_to_be_filtered(content3,"是")
                question["question"] = [question["question"][i] for i in contin]
                question["answer"] = [question["answer"][i] for i in contin]
                contins = [contins[i] for i in contin]
        
        if 4 in question_filter_choose:
            print(4)
            content4 = filter_absence(question,
                                    question_type,
                                    model,
                                    **kwargs)
            print("content")
            print(content4)
            if content4 == None:
                return False
            
            if isinstance(content4, str):
                if content4 == "否":
                    return False
                elif all(element not in [5,6] for element in question_filter_choose):
                    return True
            else:
                if any(x is None for x in content4):
                    return result
                contin = confirm_judge_to_be_filtered(content4,"否")
                question["question"] = [question["question"][i] for i in contin]
                question["answer"] = [question["answer"][i] for i in contin]
                contins = [contins[i] for i in contin]
            
        if 5 in question_filter_choose:
            content5 = filter_proper(question,
                                    question_type,
                                    model,
                                    **kwargs)
            print(content5)
            if content5 == None:
                return False

            if isinstance(content5, str):
                if content5 == "否":
                    return False
                elif all(element not in [6] for element in question_filter_choose):
                    return True
            else:
                if any(x is None for x in content5):
                    return result
                contin = confirm_judge_to_be_filtered(content5,"否")
                question["question"] = [question["question"][i] for i in contin]
                question["answer"] = [question["answer"][i] for i in contin]
                contins = [contins[i] for i in contin]
                
        if 6 in question_filter_choose:
            print(6)
            content6 = filter_compare(question_choice,
                                      question,
                                      question_type,
                                      model,
                                      **kwargs)
            print(content6)
            if content6 == None:
                return False

            if isinstance(content6, str):
                if content6 == "否":
                    return False
                else:
                    return True
            else:
                if any(x is None for x in content6):
                    return result
                contin = confirm_judge_to_be_filtered(content6,"否")
                question["question"] = [question["question"][i] for i in contin]
                question["answer"] = [question["answer"][i] for i in contin]
                contins = [contins[i] for i in contin]
        for i in contins:
            result[i] = True
        return result

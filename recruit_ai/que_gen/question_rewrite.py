import json
from autogen import AssistantAgent, UserProxyAgent,GroupChat, GroupChatManager

# Configure LLM (price field removed)
LLM_CONFIG = {
    "config_list": [{
        "model": "deepseek-chat",
        "api_key": "",
        "base_url": ""
    }]
}

class QuestionOptimizer:
    def __init__(self):
        self.output_file = "optimized_questions.json"
    
    def _load_questions(self) -> list:
        """Loading topics to be optimized"""
        with open("advanced_questions.json", "r", encoding="utf-8") as f:
            return [q for q in json.load(f) if any(t["difficulty"]=="高级" for t in q["types"].values())][:50]

    def _save_results(self, results: list):
        """Save optimization results (keep original format)"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def run_optimization(self):
        # Initialize the Agent Team
        evaluator = AssistantAgent(
            name="Evaluator",
            system_message="""作为招聘领域题目评估专家，请检查题目是否满足以下条件：
1. 是否考察实际问题解决能力
2. 是否结合理论实践
3. 是否存在表述问题
4. 参考答案是否匹配
5. 术语是否规范
如果满足以上条件，请回复APPROVE；如果不满足以上条件，请给出优化建议
""",
#Return {decision: APPROVE/REVISE, feedback: improvement suggestions} in JSON format""",
            llm_config=LLM_CONFIG
        )

        optimizer = AssistantAgent(
            name="Optimizer",
            system_message="""根据评估建议优化题目，保持原题目的JSON结构，直接输出修改后的完整题目,不需要再输出优化说明，题目格式如下：
{
    "id": 11,
    "field": "视觉/交互/设计",
    "label": [
        "视觉设计"
    ],
    "types": {
        "choice": {
            "question": "iPhone通讯录中的手机号码被分割成”xxx-xxx-xxxx-xxxx“的形式，运用（）法则的小细节设计，用来减轻用户记忆负担。",
            "option": [
                "8±3",
                "11-7",
                "7±2",
                "3+4+4"
            ],
            "answer": "C",
            "difficulty": "高级"
        }
    }
}""",
            llm_config=LLM_CONFIG
        )

        user_proxy = UserProxyAgent(
            name="User_Proxy",
            is_termination_msg=lambda msg: "APPROVE" in msg.get("content", ""),
            human_input_mode="NEVER",
            code_execution_config=False
        )

        # Define the termination condition: terminate when Evaluator returns "APPROVE"
        #termination_condition = TextMentionTermination("APPROVE")

        #Configuring group chat (single-turn conversation)
        group_chat = GroupChat(
            agents=[user_proxy, evaluator, optimizer],
            messages=[], 
            max_round=6,
            speaker_selection_method="round_robin")  # Binding termination conditions

        manager = GroupChatManager(groupchat=group_chat)
        
        results = []

        for original_q in self._load_questions():
            user_proxy.initiate_chat(
                manager,
                message=f"对以下题目提出优化建议：{json.dumps(original_q, ensure_ascii=False)}",
                clear_history=True  # Key parameter: Force cleanup history
            )
            
            # Extract the final optimization results
            #final_msg = optimizer.last_message(manager)
            final_msg = optimizer.last_message(manager)

            user_proxy.reset()
            manager.reset()
            #group_chat.reset()

            try:
                optimized = final_msg
                if isinstance(optimized, dict) and "id" in optimized:
                    results.append(optimized)  # Directly store optimized questions
                else:
                    results.append(final_msg)
            except json.JSONDecodeError as e:
                results.append({
                    "error": f"JSON解析失败: {str(e)}",
                    "original_id": original_q.get("id"),
                    "raw_output": final_msg
                })
        
        self._save_results(results)

if __name__ == "__main__":
    optimizer = QuestionOptimizer()
    optimizer.run_optimization()
    print(f"优化结果已保存至 {optimizer.output_file}")

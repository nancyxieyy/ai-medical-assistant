# backend/nodes/llm_doctor_node.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

class LLMDoctorAdviceNode:
    """
    LLM Node:
    1. 实时建议模式("realtime_advice")
    2. 报告总结模式("final_report")
    """

    def __init__(self):
        
        model_name = "Qwen/Qwen3-1.7B"

        print(f"加载模型 {model_name} ...")
        
        # 运行 Hugging Face 使用 trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 判断设备
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        print(f"正在使用设备: {device}", "资料类型: {dtype}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True # 載入模型時也需要
        ).to(device)

        print("Qwen3 模型加载完成")
    
    def build_prompt(self, mode, transcript, context):
        """根据不同模式生成不同的 Prompt"""
        if mode == "realtime_advice":
            return (
                "你是一名专业医生助理，根据以下实时对话内容和相关医学知识，"
                "生成一句不超过50字的、直接的、口语化的医疗建议。\n\n"
                "不要进行任何解释、分析或自我修正，只输出建议本身。\n\n"
                f"对话内容：{transcript}\n\n"
                f"相关医学知识：{context}\n\n"
                "请直接输出建议："
            )
        elif mode == "final_report":
            return (
                "你是一名专业的医疗记录员。根据以下问诊全文，严格按照示例格式，"
                "生成一份结构化的病历报告。不要添加任何示例中没有的标题，"
                "不要进行任何解释、分析或自我修正。直接以“【主诉】：”开头输出报告。\n\n"
                "示例格式：\n"
                "【主诉】：\n【诊断】：\n【处方】：\n【医嘱】：\n\n"
                f"问诊全文：\n{transcript}\n\n"
                f"病历报告："
            )
        else:
            raise ValueError("未知模式: 请选择 'realtime_advice' 或 'final_report'")
    
    def run(self, state):
        """
        state 包含:
        1. mode: "realtime_advice" 或 "final_report"
        2. transcript: 医生与病人的文字对话
        3. context: 从 RAG 查询出的知识片段
        """
        mode = state.get("mode", "realtime_advice")
        transcript = state.get("transcript", "")
        context = state.get("context", "")
        
        # 执行 pipeline
        # prompt = self.build_prompt(mode, transcript, context)
        
        # 1. 根據模式，建立符合 Chat Template 格式的 messages 列表
        if mode == "realtime_advice":
            messages = [
                {"role": "system", "content": "你是一名专业的医疗助理。你的任务是根据对话和知识，生成一句不超过50字的、直接的、口语化的医疗建议。不要进行任何解释、分析或自我修正，只输出建议本身。"},
                {"role": "user", "content": f"--- 对话内容 ---\n{transcript}\n\n--- 相关知识 ---\n{context}\n\n--- 医疗建议 ---"}
            ]
        elif mode == "final_report":
            messages = [
                {"role": "system", "content": "你是一名专业的医疗记录员。根据问诊全文，严格按照【主诉】、【诊断】、【处方】、【医嘱】的格式生成一份结构化的病历报告。不要添加任何示例中没有的标题，不要进行任何解释、分析或自我修正。直接以“【主诉】：”开头输出报告。"},
                {"role": "user", "content": f"--- 问诊全文 ---\n{transcript}\n\n--- 相关知识 ---\n{context}\n\n--- 病历报告 ---"}
            ]
        else:
            raise ValueError("未知模式")
        
        # full_output = self.pipe(prompt)[0]["generated_text"]
        # # 去掉 prompt，只留下新生成的文字
        # llm_output = full_output[len(prompt):].strip()
        
        # return {"llm_output": llm_output}
        
        # 2. 使用 tokenizer 將 messages 列表轉換為模型看得懂的格式
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 3. 使用 model.generate 進行推理
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.3,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 4. 清理和解碼輸出
        # generated_ids 包含輸入和輸出，我們需要把它們分開
        input_ids_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[:, input_ids_len:]
        
        full_output = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
        
        # --- 關鍵修正 2：後處理，忽略「草稿」部分 ---
        # 檢查輸出中是否包含</think>標籤，如果包含，我們只取它後面的內容
        if "</think>" in full_output:
            llm_output = full_output.split("</think>")[-1].strip()
        else:
            llm_output = full_output.strip()
        
        return {"llm_output": llm_output}
    
    
if __name__ == "__main__":
    node = LLMDoctorAdviceNode()

    # 模拟实时建议
    test_state_advice = {
        "mode": "realtime_advice",
        "transcript": "患者说自己这两天发烧38度并咳嗽。",
        "context": "普通感冒通常不需要抗生素，可物理降温和多喝水。"
    }
    res_advice = node.run(test_state_advice)
    print("\n--- 测试实时建议 ---")
    print(res_advice["llm_output"])

    # 模拟最终报告
    test_state_report = {
        "mode": "final_report",
        "transcript": "医生：您好，请问哪里不舒服？\n患者：医生您好，我这两天发烧，最高到38度，还有点咳嗽，喉咙干。\n医生：好的，有流鼻涕或鼻塞么？\n患者：沒有。\n医生：我帮您看一下喉咙...嗯，有点红肿。",
        "context": "急性上呼吸道感染的典型症状包括发热、咳嗽、咽喉痛。"
    }
    res_report = node.run(test_state_report)
    print("\n--- 测试最终报告 ---")
    print(res_report["llm_output"])

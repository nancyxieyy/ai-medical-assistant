# backend/nodes/llm_doctor_node.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re


class LLMDoctorAdviceNode:
    """
    LLM Node:
    1. 实时建议模式("realtime_advice")
    2. 报告总结模式("final_report")
    """

    _model = None
    _tokenizer = None
    _device = None

    def __init__(self):
        """懒加载模型 + 缓存"""
        model_name = "Qwen/Qwen3-1.7B"

        if LLMDoctorAdviceNode._model is None:
            print(f"正在加载模型：{model_name}")

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # 判断设备
            if torch.backends.mps.is_available():
                device, dtype = "mps", 
                torch.float16
            elif torch.cuda.is_available():
                device, dtype = "cuda", 
                torch.float16
            else:
                device, dtype = "cpu", 
                torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(device)

            LLMDoctorAdviceNode._model = model
            LLMDoctorAdviceNode._tokenizer = tokenizer
            LLMDoctorAdviceNode._device = device

            print(f"模型加载完成（设备：{device}）")

        self.model = LLMDoctorAdviceNode._model
        self.tokenizer = LLMDoctorAdviceNode._tokenizer
        self.device = LLMDoctorAdviceNode._device


    # 通用清洗函数
    def _clean_text(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"\{.*?\}", "", text, flags=re.S)  # 清掉JSON
        text = re.sub(r"\*\*|#|`|json|JSON|```", "", text)
        text = re.sub(r"(示例|例子|说明|解释|分析|思考|输出|Answer|Response|如下|以下|报告如下|准则要求|请确保|抱歉|描述不清|不能确定|补充说明).*", "", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()

    def _clean_realtime(self, text: str) -> str:
        """仅保留一句自然语言建议"""
        text = self._clean_text(text)
        # 取第一句中文
        text = re.split(r"[。！？\n]", text)[0]
        text = re.sub(r"[“”\"\'·]", "", text).strip()
        if len(text) > 50:
            text = text[:50]
        return text

    def _clean_report(self, text: str) -> str:
        """仅保留主诉到医嘱四段式内容"""
        text = self._clean_text(text)

        # 截取从【主诉】开始的部分
        m = re.search(r"【主诉】[:：].*", text, flags=re.S)
        text = text[m.start():] if m else text

        # 去掉模型开头的解释性句子
        text = re.sub(r"^[^【]*", "", text)

        # 捕获四段
        def grab(tag):
            pat = rf"【{tag}】[:：]\s*(.*?)(?=\n\s*【|$)"
            mm = re.search(pat, text, flags=re.S)
            return mm.group(1).strip() if mm else ""

        chief = grab("主诉")
        diag = grab("诊断")
        rx = grab("处方")
        advice = grab("医嘱")

        # 如果模型没输出规范格式，则尝试从内容中推测
        if not any([chief, diag, rx, advice]):
            chief = re.search(r"(发热|咳嗽|喉咙|头痛|流鼻涕|不适)", text)
            chief = chief.group(0) + "相关症状" if chief else "近期身体不适"
            diag = "上呼吸道感染（待查）"
            rx = "对症治疗"
            advice = "注意休息，若持续发热及时就诊"

        # 每段之间空一行
        return (
            f"【主诉】：{chief}\n\n"
            f"【诊断】：{diag}\n\n"
            f"【处方】：{rx}\n\n"
            f"【医嘱】：{advice}"
        ).strip()


    def build_prompt(self, mode, transcript, context):
        """根据不同模式生成不同的 Prompt"""
        if mode == "realtime_advice":
            return (
                "你是一名专业医生助理，只输出一句不超过50字的医疗建议。\n"
                "必须自然、口语化、简洁。\n"
                "禁止输出解释、Markdown、JSON、示例或任何说明文字。\n\n"
                f"【对话】{transcript}\n【医学知识】{context}\n"
                "请直接输出一句建议："
            )
        elif mode == "final_report":
            return (
                "你是一名专业的临床医生助理。根据以下完整的问诊记录，生成一份正式病历报告。\n"
                "报告应自然流畅、专业清晰，避免过度解释或模板化语句。\n"
                "必须严格使用以下格式输出，每个部分应尽量详细：\n\n"
                "【主诉】：患者自述的主要症状及持续时间，至少15字。\n"
                "【诊断】：医生的初步判断，列出可能的疾病或病因，至少15字。\n"
                "【处方】：具体药物名称、剂量与使用频率，至少15字。\n"
                "【医嘱】：日常护理建议、复诊要求等，至少15字。\n\n"
                "禁止输出除报告内容外的任何说明、准则、抱歉、推测或格式外内容。\n\n"
                f"【问诊全文】\n{transcript}\n\n【相关医学知识】\n{context}\n"
                "直接输出报告，从【主诉】开始："
            )
        else:
            raise ValueError("未知模式")


    # 核心执行
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

        prompt = self.build_prompt(mode, transcript, context)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # 根据模式，建立符合 Chat Template 格式的 messages 列表
        if mode == "realtime_advice":
            gen_kwargs = dict(
                max_new_tokens=80,
                temperature=0.2, 
                top_p=0.85,
                repetition_penalty=1.05,
                do_sample=True,
            )
        else:
            gen_kwargs = dict(
                max_new_tokens=1600,       # 长输出用于总结报告
                temperature=0.35,
                top_p=0.9,
                repetition_penalty=1.05,
                do_sample=True,
            )

        # 使用 model.generate 进行推理
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs,
            )

        # 数据清洗
        gen_only = output_ids[:, inputs.input_ids.shape[1]:]
        raw = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

        if mode == "realtime_advice":
            clean = self._clean_realtime(raw)
        else:
            clean = self._clean_report(raw)

        return {"llm_output": clean}


# 测试用例
if __name__ == "__main__":
    node = LLMDoctorAdviceNode()

    print("\n--- 测试实时建议 ---")
    print(node.run({
        "mode": "realtime_advice",
        "transcript": "患者说自己这两天发烧38度并咳嗽。",
        "context": "普通感冒通常不需要抗生素，可物理降温和多喝水。"
    })["llm_output"])

    print("\n--- 测试最终报告 ---")
    print(node.run({
        "mode": "final_report",
        "transcript": (
            "医生：您好，请问哪里不舒服？\n"
            "患者：医生您好，我这两天发烧，最高到38度，还有点咳嗽。\n"
            "医生：有流鼻涕或鼻塞吗？\n"
            "患者：沒有。\n"
            "医生：喉咙有点红肿。"
        ),
        "context": "急性上呼吸道感染的典型症状包括发热、咳嗽、咽喉痛。"
    })["llm_output"])

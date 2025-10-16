# backend/ASR/Agent_ASR_Integration.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 说明：
# - ASR 实时识别每一句话；
# - LangGraph Realtime Agent 异步生成实时建议；
# -  自动积累全文；
# -  录音结束后由 Summary Agent 生成病历报告草稿。

import os
import sys
from pathlib import Path
import concurrent.futures
import time

os.environ["HUGGINGFACE_HUB_CACHE"] = ".cache/hf"

try:
    # 负责跨平台音频采集（此处作为麦克风输入）
    import sounddevice as sd
except ImportError:
    print("请先安装 sounddevice: pip install sounddevice -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(-1)

try:
    # sherpa-onnx 提供在线流式识别接口
    import sherpa_onnx
except ImportError:
    print("请先安装 sherpa-onnx: pip install sherpa-onnx -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(-1)


# 导入 LangGraph Agents
from backend.ASR.medical_graphs import build_realtime_agent, build_summary_agent


def create_recognizer():
    """创建 sherpa-onnx 在线识别器（Paraformer 版本）。

    注意：路径需与本地模型目录一致；当前配置启用端点检测。
    """
    encoder = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx"
    decoder = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx"
    tokens = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt"
    
    # 检查是否缺少模型
    for f in [encoder, decoder, tokens]:
        if not Path(f).is_file():
            print(f"缺少模型文件: {f}")
            print("请从 ModelScope 下载: https://modelscope.cn/models/pengzhendong/sherpa-onnx-streaming-paraformer-bilingual-zh-en")
            sys.exit(-1)

    recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        sample_rate=16000,
        feature_dim=80,
        provider="cpu",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.5,
        rule2_min_trailing_silence=2.0,
        rule3_min_utterance_length=500,
    )
    return recognizer


def main():
    recognizer = create_recognizer()
    realtime_agent = build_realtime_agent()
    summary_agent = build_summary_agent()

    print("\n开始实时问诊，请讲话...(Ctrl+C 结束录音)\n")

    transcript_all = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=48000) as s:
            stream = recognizer.create_stream()
            samples_per_read = int(0.1 * 48000)

            while True:
                samples, _ = s.read(samples_per_read)
                stream.accept_waveform(48000, samples.reshape(-1))
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                result_text = recognizer.get_result(stream)

                last_advice_time = 0
                ADVICE_COOLDOWN = 3.0
                # 检测端点（句子结束）
                if recognizer.is_endpoint(stream):
                    if result_text.strip():
                        now = time.time()
                        if now - last_advice_time > ADVICE_COOLDOWN:
                            print(f"\n识别结果：{result_text}")
                            transcript_all.append(result_text)

                            # 异步执行实时建议生成（防止阻塞音频流）
                            def async_advice(text):
                                try:
                                    out = realtime_agent.invoke({"texts": [text], "mode": "realtime_advice"})
                                    suggestion = out.get("llm_output") or out.get("output") or "无建议"
                                    if suggestion.strip() and suggestion != "无建议":
                                        print(f"实时建议：{suggestion}")
                                except Exception as e:
                                    print(f"[Agent错误] {e}")

                            executor.submit(async_advice, result_text)
                            last_advice_time = now

                    recognizer.reset(stream)

    except KeyboardInterrupt:
        print("\n录音结束，正在生成总结报告...\n")
        transcript_text = "\n".join(transcript_all)
        Path("asr_transcript.txt").write_text(transcript_text, encoding="utf-8")

        if not transcript_all:
            print("未识别到有效语音，未生成报告。")
            return

        try:
            result = summary_agent.invoke({
                "texts": [transcript_text],
                "mode": "final_report"
            })
            report = result.get("llm_output", "")
            Path("report_draft.txt").write_text(report, encoding="utf-8")

            print("\n病历报告草稿已生成：report_draft.txt\n")
            print(report)
        except Exception as e:
            print(f"[生成报告出错] {e}")

        print("\n结束。")


if __name__ == "__main__":
    main()

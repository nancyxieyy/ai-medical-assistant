#!/usr/bin/env python3
# 说明：
#  - 本脚本从麦克风实时录音，使用 sherpa-onnx 进行流式语音识别。
#  - 每当端点检测判定一句话结束且有文本结果时，将结果通过 LangGraph Agent 的 invoke 发送。
#  - Agent 此处为一个最小示例节点，接收输入后直接在控制台打印。[Agent] 收到: <文本>

import os
import sys
from pathlib import Path

try:
    # 负责跨平台音频采集（此处作为麦克风输入）
    import sounddevice as sd
except ImportError:
    print("请先安装 sounddevice: pip install sounddevice")
    sys.exit(-1)

try:
    # sherpa-onnx 提供在线流式识别接口
    import sherpa_onnx
except ImportError:
    print("请先安装 sherpa-onnx: pip install sherpa-onnx")
    sys.exit(-1)

try:
    # LangGraph：用于定义一个简单的 Agent 图
    from langgraph.graph import StateGraph, END
except ImportError:
    print("未安装 langgraph，请先运行: pip install langgraph langchain-core")
    sys.exit(-1)

from typing import TypedDict

#from display import Display  # 如需可视化历史与进行中文本，可启用相关行


class AgentState(TypedDict):
    # 定义 Agent 图的共享状态结构
    input: str
    output: str


def create_agent():
    """构建并编译一个最简 LangGraph Agent。

    该 Agent 只有一个节点：接收输入文本后直接打印并原样返回到 output。
    """
    graph = StateGraph(AgentState)

    def printer_node(state: AgentState) -> AgentState:
        # 最小打印节点：副作用是打印，返回值写回状态
        text = state.get("input", "")
        print(f"[Agent] 收到: {text}")
        return {"output": text}

    graph.add_node("printer", printer_node)
    graph.set_entry_point("printer")
    graph.add_edge("printer", END)
    return graph.compile()


def assert_file_exists(filename: str):
    """简单文件存在性校验，不存在时直接退出。"""
    if not Path(filename).is_file():
        print(f"{filename} 不存在！")
        sys.exit(-1)


def create_recognizer():
    """创建 sherpa-onnx 在线识别器（Paraformer 版本）。

    注意：路径需与本地模型目录一致；当前配置启用端点检测。
    """
    encoder = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx"
    decoder = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx"
    tokens = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt"
    assert_file_exists(encoder)
    assert_file_exists(decoder)
    assert_file_exists(tokens)
    recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        num_threads=100,
        sample_rate=16000,
        feature_dim=80,
        provider="cuda",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        debug=1,# 基本等于关闭该规则
    )
    return recognizer


def main():
    # 查询本机音频设备；无设备则退出
    devices = sd.query_devices()
    if len(devices) == 0:
        print("未找到麦克风设备")
        sys.exit(0)

    # 输出默认输入设备名称
    default_input_device_idx = sd.default.device[0]
    print(f"使用默认输入设备: {devices[default_input_device_idx]['name']}")

    # 初始化识别器与 Agent
    recognizer = create_recognizer()
    agent = create_agent()
    # 如需在控制台显示进行中与历史文本，取消以下行注释并启用对应调用

    print("开始！请讲话... (Ctrl+C 退出)")

    # 麦克风采样率 48k，sherpa-onnx 内部会重采样到模型采样率 16k
    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 每次读取 100ms 音频

    stream = recognizer.create_stream()

    # 打开输入流，循环读取音频 -> 解码 -> 端点判断
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            is_endpoint = recognizer.is_endpoint(stream)
            result_text = recognizer.get_result(stream)

            if is_endpoint:
                # 端点触发（判定一句话结束）且有文本，调用 Agent
                if result_text:
                    try:
                        agent.invoke({"input": result_text})
                    except Exception as e:
                        print(f"[Agent 错误] {e}")
                # 重置流以开始下一句
                recognizer.reset(stream)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已捕获 Ctrl + C，正在退出")



# scripts/test_asr.py

# from funasr import AutoModel
# import soundfile

# # 模型初始化
# print("正在加载 FunASR 模型，请稍候...")
# model = AutoModel(
#     model="paraformer-zh-streaming", model_revision="v2.0.4",
#     # vad_model="fsmn-vad", vad_model_revision="v2.0.4",
#     vad_model=None, 
#     punc_model="ct-punc-c", punc_model_revision="v2.0.4",
#     spk_model="cam++", spk_model_revision="v2.0.2"
#     )
# print("模型加载完成，正在进行语音识别...")

# # 准备音频和流式参数
# audio_file_path = "scripts/audio/2speakers_example.wav"

# # 流式识别参数
# chunk_size = [0, 10, 5]
# chunk_stride = chunk_size[1] * 960

# # 手动模拟流式读取与识别
# # 读取音频文件
# try:
#     speech, sample_rate = soundfile.read(audio_file_path)
#     print(f"成功读取音频文件: {audio_file_path}")
# except Exception as e:
#     print(f"读取音频文件时出错: {e}")
#     exit(1)

# # 初始化一个空的 cache， 用来存储流式识别的中间状态
# cache = {}
# total_chunk_num = -(-(len(speech) - 1) // chunk_stride)

# print(f"音频被切分为 {total_chunk_num} 个片段进行流式识别...")
# print("识别结果：")


# # 收集所有片段的文本结果
# transcript_segments = []

# for i in range(total_chunk_num):
#     start = i * chunk_stride
#     end = start + chunk_stride
#     speech_chunk = speech[start:end]
#     is_final = (i == total_chunk_num - 1)
    
#     # 调用模型进行流式识别
#     result = model.generate(
#         input=speech_chunk, 
#         cache=cache, 
#         is_final=is_final
#     )
    
#     # 收集所有文本
#     if result and len(result) > 0 and result[0].get('text') and 'spk' in result[0]:
#         transcript_segments.append({
#             "speaker": result[0]['spk'],
#             "text": result[0]['text']
#         })
#         print(".", end="", flush=True) # 打印进度点

# # 打印最终结果
# print("\n\n最终识别结果：")
# if not transcript_segments:
#     print("未识别到任何语音内容。")
# else:
#     current_speaker = None
#     # 逐段打印，区分说话人
#     for i, segment in enumerate(transcript_segments):
#         speaker_id = segment['speaker']
#         text = segment['text']

#         # 检测说话人变化
#         if speaker_id != current_speaker:
#             current_speaker = speaker_id
#             # 换行并打印说话人标签
#             if i > 0:
#                 print()
#             print(f"說話人_{current_speaker}: ", end="")
        
#         print(text, end="")
#     print() # 最后换行

from funasr import AutoModel
import soundfile
import numpy as np
import time

print("正在加载 FunASR 模型，请稍候...")

# 1) 流式 ASR（禁用 VAD，按你的效果更好）
asr_model = AutoModel(
    model="paraformer-zh-streaming", model_revision="v2.0.4",
    vad_model=None,
    punc_model="ct-punc-c", punc_model_revision="v2.0.4"
)

# 2) 说话人特征（CAM++）
spk_model = AutoModel(
    model="cam++", model_revision="v2.0.2"
)

print("模型加载完成，正在进行语音识别...")

# ==== 音频 ====
audio_file = "scripts/audio/2speakers_example.wav"
speech, sr = soundfile.read(audio_file)
print(f"成功读取音频文件: {audio_file} (采样率: {sr})")

# ==== 流式参数 ====
# 约 0.6s/块（10*960/16000 ≈ 0.6s）
chunk_size = [0, 10, 5]
chunk_stride = chunk_size[1] * 960
total_chunk_num = -(-(len(speech) - 1) // chunk_stride)

# ==== 聚类窗口（若干块合并后再提 embedding）====
EMB_WIN_CHUNKS = 6     # 每 6 块 ≈ 3.6 秒 提一次 embedding（可调大/小）
N_SPEAKERS = 2         # 预计说话人人数（可改）

# ==== 状态 ====
cache = {}
buffer_chunks = []     # 累计若干块，用来提一次 embedding
buffer_texts = []      # 与 buffer_chunks 对应的文本

# 最终用于聚类的“段”
segments = []          # 每个元素: {"text": str, "emb_idx": int or None}
speaker_embeddings = []  # 每个有效段一条 embedding（与 emb_idx 对应）

print(f"音频被切分为 {total_chunk_num} 个片段进行流式识别...")
print("识别结果（实时显示）：")

def flush_buffer(force=False):
    """把当前缓存的若干块合并为一个段，提取 embedding（可能失败），记录到 segments。"""
    global buffer_chunks, buffer_texts, speaker_embeddings, segments

    if not buffer_chunks and not force:
        return

    # 合并文本
    seg_text = "".join(buffer_texts).strip()
    seg_has_audio = len(buffer_chunks) > 0
    emb_idx = None

    if seg_has_audio and seg_text:
        long_chunk = np.concatenate(buffer_chunks)

        # 试图提取 embedding（可能因太短/静音失败）
        try:
            spk_result = spk_model.generate(input=long_chunk)
            if isinstance(spk_result, list) and len(spk_result) > 0 and "spk_embedding" in spk_result[0]:
                emb = np.asarray(spk_result[0]["spk_embedding"])
                speaker_embeddings.append(emb)
                emb_idx = len(speaker_embeddings) - 1
            else:
                # 提取失败，保留文本，emb_idx 留空
                pass
        except Exception as e:
            # 提取异常，也保留文本
            pass

        segments.append({"text": seg_text, "emb_idx": emb_idx})

    # 清空缓冲
    buffer_chunks = []
    buffer_texts = []

# ==== 主循环：实时识别 + 缓存 ====
for i in range(total_chunk_num):
    start = i * chunk_stride
    end = start + chunk_stride
    chunk = speech[start:end]
    is_final = (i == total_chunk_num - 1)

    # 实时 ASR
    result = asr_model.generate(input=chunk, cache=cache, is_final=is_final)

    if result and len(result) > 0 and result[0].get("text"):
        text = result[0]["text"]
        # 实时显示
        print(f"[实时识别] {text}")
        # 累计到窗口
        buffer_chunks.append(chunk)
        buffer_texts.append(text)

        # 到达窗口大小就冲洗一次
        if len(buffer_chunks) >= EMB_WIN_CHUNKS:
            flush_buffer()
    # 小睡，模拟 UI 节奏（可去掉）
    time.sleep(0.05)

# 循环结束，强制冲洗缓冲
flush_buffer(force=True)

# ==== 聚类 ====
if len(speaker_embeddings) == 0:
    print("\n 最终未提取到任何说话人 embedding。")
    print("建议：增大 EMB_WIN_CHUNKS（例如 8~12），或确保段落里确实有人声。")
    # 即便没有 embedding，也把文本打印出来，标记 Unknown
    print("\n===== 最终区分说话人的识别结果（无embedding，标为 Unknown） =====")
    for seg in segments:
        if seg["text"]:
            print(f"Speaker_?: {seg['text']}")
    print("\n识别完成。")
    raise SystemExit(0)

# 用 sklearn 做聚类
try:
    from sklearn.cluster import AgglomerativeClustering
    embeddings_mat = np.vstack(speaker_embeddings)
    try:
        clustering = AgglomerativeClustering(n_clusters=N_SPEAKERS, metric='cosine', linkage='average')
    except TypeError:
        # 兼容旧版 sklearn
        clustering = AgglomerativeClustering(n_clusters=N_SPEAKERS, affinity='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings_mat)
except Exception as e:
    print(f"\n⚠️ 聚类出错：{e}")
    print("仅输出未分配说话人的文本。")
    labels = None

# ==== 打印最终（为无 embedding 的段继承最近标签）====
print("\n===== 最终区分说话人的识别结果 =====")
last_label = 0
for seg in segments:
    text = seg["text"]
    if not text:
        continue
    if seg["emb_idx"] is not None and labels is not None:
        label = int(labels[seg["emb_idx"]])
        last_label = label
    else:
        # 没有 embedding 的段，继承最近一次成功的标签
        label = last_label
    print(f"Speaker_{label+1}: {text}")

print("\n识别完成。")
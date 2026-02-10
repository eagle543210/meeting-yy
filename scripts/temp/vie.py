import os
import uuid
import time
import asyncio
import sounddevice as sd
import numpy as np
import torch
from queue import Queue
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Inference

# =========================================================
# 配置文件参数
# =========================================================

# Milvus 连接参数
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "default")

# Milvus 集合参数
MILVUS_DIMENSION = 512
MILVUS_VOICE_COLLECTION_NAME = "voice_prints"
VOICEPRINT_SIMILARITY_THRESHOLD = 0.8  # 用于判断是否为同一人

# 语音处理参数
MIN_SPEECH_SEGMENT_DURATION = 1.5
AUDIO_ENERGY_THRESHOLD = 0.02 # 降低阈值以增加敏感度
MIN_SPEECH_DURATION_OFF = 0.8
PYANNOTE_EMBEDDING_MODEL = "pyannote/embedding@2.1"

# =========================================================
# 服务类：用于与 Milvus 和声纹模型交互 (代码与上一次提供的一致)
# =========================================================

class MilvusService:
    # ... 保持不变 ...
    def __init__(self):
        self.collection = None
        
    def connect(self):
        try:
            connections.connect(
                alias=MILVUS_ALIAS,
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                user=MILVUS_USER,
                password=MILVUS_PASSWORD,
            )
            print("✅ 成功连接到 Milvus 服务。")
        except Exception as e:
            print(f"❌ 无法连接到 Milvus：{e}")
            exit()

    def create_collection(self):
        if utility.has_collection(MILVUS_VOICE_COLLECTION_NAME, using=MILVUS_ALIAS):
            self.collection = Collection(MILVUS_VOICE_COLLECTION_NAME, using=MILVUS_ALIAS)
            print(f"✅ 集合 '{MILVUS_VOICE_COLLECTION_NAME}' 已存在，正在加载...")
            self.collection.load()
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
                FieldSchema(name="user_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=MILVUS_DIMENSION)
            ]
            schema = CollectionSchema(fields, "声纹特征集合")
            self.collection = Collection(MILVUS_VOICE_COLLECTION_NAME, schema=schema, using=MILVUS_ALIAS)

            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
            print(f"✅ 新的集合 '{MILVUS_VOICE_COLLECTION_NAME}' 和索引已创建并加载。")

    def search_voiceprint(self, embedding: np.ndarray):
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}
        
        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=1,
            output_fields=["user_name", "role"]
        )
        return results

    def insert_voiceprint(self, user_id: str, user_name: str, embedding: np.ndarray):
        data = [[user_id], [user_name], ["GUEST"], [embedding.tolist()]]
        self.collection.insert(data)
        self.collection.flush()

class VoiceprintService:
    # ... 保持不变 ...
    def __init__(self, milvus_service: MilvusService):
        self.milvus_service = milvus_service
        self.embedding_model = Inference(PYANNOTE_EMBEDDING_MODEL)

    def extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
        
        audio_pyannote = SlidingWindowFeature(audio_data.reshape(-1, 1), SlidingWindow(0, 1/16000))
        
        embedding = self.embedding_model(audio_pyannote)
        
        return embedding.squeeze().numpy()

    async def identify_or_register_speaker(self, audio_data: np.ndarray) -> dict:
        print("\n--- 正在执行声纹识别与注册逻辑 ---")
        
        if len(audio_data) / 16000 < MIN_SPEECH_SEGMENT_DURATION:
            print("❌ 语音片段太短，不满足声纹识别的最低时长要求。")
            return {"status": "too_short", "message": "语音片段太短"}
        
        voice_embedding = self.extract_embedding(audio_data)
        
        print("正在 Milvus 中查找匹配声纹...")
        search_results = self.milvus_service.search_voiceprint(voice_embedding)
        
        threshold_for_milvus_distance = 1 - VOICEPRINT_SIMILARITY_THRESHOLD
        
        if search_results and search_results[0][0].distance < threshold_for_milvus_distance:
            hit = search_results[0][0]
            matched_id = hit.id
            matched_name = hit.entity.get("user_name")
            print(f"🚀 识别成功！匹配到用户: {matched_name} (ID: {matched_id}), 距离: {hit.distance:.4f}")
            return {"status": "recognized", "user_id": matched_id, "user_name": matched_name}
        
        else:
            new_user_id = str(uuid.uuid4())
            new_username = f"新用户_{len(self.milvus_service.collection.entities) + 1}"
            
            self.milvus_service.insert_voiceprint(new_user_id, new_username, voice_embedding)
            print(f"📝 未找到匹配，已注册新用户: {new_username} (ID: {new_user_id})")
            return {"status": "registered", "user_id": new_user_id, "user_name": new_username}

# =========================================================
# 主程序：持续监听并处理
# =========================================================

# 全局变量
audio_queue = Queue()

def audio_callback(indata, frames, time, status):
    """
    当有新的音频数据时，sounddevice 会调用此函数。
    它将音频块放入队列中。
    """
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

async def process_audio_segments(voiceprint_service):
    """
    异步任务，持续从队列中取出音频数据并处理。
    """
    audio_buffer = np.array([])
    speaking = False
    silence_start_time = 0

    while True:
        try:
            # 从队列中获取音频块
            audio_chunk = audio_queue.get_nowait()
            audio_data = audio_chunk.squeeze()
            
            # 计算音频能量 (RMS)
            rms = np.sqrt(np.mean(audio_data**2))
            
            is_speech = rms > AUDIO_ENERGY_THRESHOLD
            
            if is_speech:
                if not speaking:
                    print("\n🎙️ 检测到语音活动，开始录音...")
                    speaking = True
                audio_buffer = np.concatenate((audio_buffer, audio_data))
                silence_start_time = 0
            
            elif speaking:
                # 处于静音状态
                if silence_start_time == 0:
                    silence_start_time = time.time()
                
                # 如果静音持续时间超过阈值，则认为一个语音片段结束
                if time.time() - silence_start_time > MIN_SPEECH_DURATION_OFF:
                    print(f"静音时间超过 {MIN_SPEECH_DURATION_OFF} 秒，处理语音片段...")
                    speaking = False
                    
                    if len(audio_buffer) / 16000 >= MIN_SPEECH_SEGMENT_DURATION:
                        await voiceprint_service.identify_or_register_speaker(audio_buffer)
                    else:
                        print("❌ 语音片段太短，不满足处理要求。")
                    
                    # 清空缓冲区
                    audio_buffer = np.array([])
        
        except asyncio.QueueEmpty:
            # 当队列为空时，不做任何处理，让循环继续。
            # 这里的 pass 是为了避免捕获 Empty 异常后，被另一个异常处理块处理。
            await asyncio.sleep(0.01) # 短暂休眠，避免 CPU 占用过高
        except Exception as e:
            # 仅捕获真正的意外错误
            print(f"处理音频时发生意外错误: {e}")
            audio_buffer = np.array([])
            speaking = False
            
async def main():
    fs = 16000 # 采样率
    
    milvus_service = MilvusService()
    milvus_service.connect()
    milvus_service.create_collection()
    
    voiceprint_service = VoiceprintService(milvus_service)
    
    try:
        print("🚀 程序已启动，正在持续监听麦克风。请开始说话...")
        
        # 启动音频输入流
        with sd.InputStream(samplerate=fs, channels=1, callback=audio_callback, dtype='float32'):
            # 启动音频处理任务
            await process_audio_segments(voiceprint_service)
            
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("--- 详细错误信息 ---")
        traceback.print_exc()  # <-- 新增这一行
        print("--------------------")
        
        print("请检查你的麦克风、Milvus 配置或 Hugging Face 认证是否正确。")
    finally:
        if connections.has_connection(MILVUS_ALIAS):
            connections.disconnect(MILVUS_ALIAS)
            print("✅ 已断开 Milvus 连接。")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
# M:\meeting\config\settings.py

import os
from typing import Optional, ClassVar, Dict, Any, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
import torch 
from pathlib import Path 
import logging

# 导入 Milvus 相关的 DataType，用于定义 Schema
from pymilvus import DataType, FieldSchema

logger = logging.getLogger(__name__)

# --- 路径配置 (全局定义，在类定义定义之前) ---
# 项目根目录，根据当前文件路径自动推断
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 所有模型文件的基础存储目录
MODEL_BASE_DIR: str = os.path.join(PROJECT_ROOT, "models")

class Settings(BaseSettings):
    """
    应用程序的配置设置。
    通过环境变量、.env 文件和默认值加载。
    """

    # Pydantic Settings 配置
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略未声明的环境变量
        case_sensitive=False   # 环境变量名称不区分大小写
    )

    # 项目根目录，用于内部 Path 构建
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # ==================== 通用设置 ====================
    APP_NAME: str = "SmartMeetingAssistant"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # ==================== 日志设置 ====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING") # 默认 INFO

    # ==================== Hugging Face Hub 设置 ====================
    # Hugging Face Hub 访问令牌 (如果需要在线下载模型，或者 pyannote.audio 需要)
    # 建议将此令牌存储在 .env 文件中，而不是直接硬编码
    
    HF_TOKEN: Optional[SecretStr] = Field(os.getenv("HF_TOKEN"), description="Hugging Face Hub 访问令牌")
    # 强制离线模式：设置为 "1" 强制离线，"0" 允许在线。这需要在导入相关库前设置。
    HF_HUB_OFFLINE: str = os.getenv("HF_HUB_OFFLINE", "1") 

    # ==================== 系统核心行为配置 ====================
    WEBSOCKET_PING_INTERVAL: ClassVar[int] = 60 # WebSocket 连接的 ping 间隔（秒）
    MONITOR_BROADCAST_INTERVAL: ClassVar[int] = 15 # 监控数据广播间隔（秒）
    USE_CUDA: bool = torch.cuda.is_available() # 是否尝试使用 CUDA (GPU) 加载模型

    # ==================== 音频处理与声纹识别配置 ====================
    VOICE_SAMPLE_RATE: int = int(os.getenv("VOICE_SAMPLE_RATE", 16000)) # 原始输入音频的采样率 (例如，从客户端接收的音频)
    VOICE_EMBEDDING_DIM: int = int(os.getenv("VOICE_EMBEDDING_DIM", 512)) # 声纹嵌入的维度，通常由声纹模型决定 (例如 ECAPA-TDNN)
    
    # VAD (语音活动检测) 模型路径
    VAD_MODEL_PATH: Path = Path(os.path.join(MODEL_BASE_DIR, "silero_vad", "silero_vad.jit")) 
    VAD_SPEECH_THRESHOLD: float = float(os.getenv("VAD_SPEECH_THRESHOLD", 0.05)) # 降低门限，提高灵敏度
    MIN_SPEECH_DURATION_OFF: float = 0.8  # 语音停顿的最小持续时间
    VAD_PAUSE_DURATION_S: float = float(os.getenv("VAD_PAUSE_DURATION_S", 1.0)) # 缩短静音判定时间
    PYANNOTE_MIN_SPEECH_DURATION_S: float = 1 # Pyannote 说话人分离模型的最小语音段持续时间 (秒)
    REALTIME_SLIDING_WINDOW_S: float = 5.0 # 滑动窗口时长，单位秒
    # 声纹识别缓冲区配置：当累积到这个时长时，才进行声纹提取。
    VOICE_BUFFER_DURATION_SECONDS: float = float(os.getenv("VOICE_BUFFER_DURATION_SECONDS", 3)) 
    # 声纹注册和识别所需的最小音频时长（秒）
    MIN_SPEECH_SEGMENT_DURATION: float = float(os.getenv("MIN_SPEECH_SEGMENT_DURATION", 1.5)) 
    # 音频能量阈值，低于此值视为静音
    AUDIO_ENERGY_THRESHOLD: float = float(os.getenv("AUDIO_ENERGY_THRESHOLD", 0.05))
    # 用于声纹嵌入模型提取声纹特征的最小语音片段时长（秒）。
    VOICE_EMBEDDING_MIN_DURATION: float = float(os.getenv("VOICE_EMBEDDING_MIN_DURATION", 1.5))

    # ==================== 模型配置 (STT, Embedding, LLM, Summary) ====================
    ENABLE_STT: bool = os.getenv("ENABLE_STT", "True").lower() == "true" # 是否启用语音转文本功能
   
    # STT (语音转文本) 模型 (Faster Whisper)
    STT_ENABLE_CONFIDENCE_FILTER: bool = False
    STT_CONFIDENCE_THRESHOLD: float = 0.3
    STT_TEMPERATURE: float = float(os.getenv("STT_TEMPERATURE", 0.0))
    STT_BEST_OF: int = int(os.getenv("STT_BEST_OF", 5))
    STT_PATIENCE: float = float(os.getenv("STT_PATIENCE", 1.0))
    WHISPER_MODEL_NAME: str = os.getenv("WHISPER_MODEL_NAME", "base") 
    WHISPER_MODEL_PATH: Path = Path(os.path.join(MODEL_BASE_DIR, WHISPER_MODEL_NAME)) 
    STT_COMPUTE_TYPE: str = os.getenv("STT_COMPUTE_TYPE", "int8") 
    STT_BEAM_SIZE: int = int(os.getenv("STT_BEAM_SIZE", 5)) 
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "zh") 
    WHISPER_MODEL_DEVICE: str = os.getenv("WHISPER_MODEL_DEVICE", "cpu")
    # Speaker Diarization (说话人分离) 模型 (pyannote.audio)
    PYANNOTE_DIARIZATION_MODEL: str = Field( 
        "pyannote/speaker-diarization-3.1",
        alias="PYANNOTE_DIARIZATION_MODEL",
        description="Pyannote 说话人分离模型的 Hugging Face Hub 名称。"
    )

    # Speaker Embedding (声纹嵌入) 模型 (SpeechBrain ECAPA-TDNN)
    PYANNOTE_EMBEDDING_MODEL: str = Field( 
        "pyannote/embedding@2.1", 
        alias="PYANNOTE_EMBEDDING_MODEL",
        description="Pyannote 声纹嵌入模型的 Hugging Face Hub 名称。"
    )
    ECAPA_MODEL_DIR: Path = Path(os.path.join(MODEL_BASE_DIR, "ecapa_tdnn")) 
    
    # LLM (大语言模型) 配置 (InternLM2.5 - GGUF 格式)
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "internlm2_5-1_8b-chat-q4_k_m.gguf") 
    LLM_MODEL_PATH: Path = Path(os.path.join(MODEL_BASE_DIR, "gguf", LLM_MODEL_NAME)) 
    # llama_cpp 相关的配置参数
    LLAMA_N_GPU_LAYERS: int = int(os.getenv("LLAMA_N_GPU_LAYERS", 0)) 
    LLAMA_N_CTX: int = int(os.getenv("LLAMA_N_CTX", 4096)) 
    
    # LLM 并发限制 (默认: 1, 适合单GPU/单机场景)
    LLM_CONCURRENCY_LIMIT: int = int(os.getenv("LLM_CONCURRENCY_LIMIT", 1))
    
     # --- RAG (检索增强生成) 配置 ---
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", 8)) 
    
    # BGE Embedding (文本嵌入) 模型
    BGE_MODEL_NAME: str = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-zh-v1.5") 
    BGE_MODEL_PATH: Path = Path(os.path.join(MODEL_BASE_DIR, "bge", "small")) 
    
    # Summary (摘要) 模型 (例如 BART)
    SUMMARY_MODEL_HUB_NAME: str = os.getenv("SUMMARY_MODEL_HUB_NAME", "facebook/bart-large-cnn")
    SUMMARY_MODEL_PATH: Path = Path(os.path.join(MODEL_BASE_DIR, "bart-large-cnn"))
    BART_NUM_BEAMS: int = int(os.getenv("BART_NUM_BEAMS", 4)) # Bart 模型生成时的 beam search 数量
    BART_LENGTH_PENALTY: float = float(os.getenv("BART_LENGTH_PENALTY", 2.0)) # Bart 模型生成时的长度惩罚
    BART_TEMPERATURE: float = float(os.getenv("BART_TEMPERATURE", 1.0)) # Bart 模型生成时的采样温度，控制多样性
    BART_MAX_INPUT_LENGTH: int = int(os.getenv("BART_MAX_INPUT_LENGTH", 1024)) # Bart 模型最大输入长度
    
    # 实时摘要的 BART 模型生成参数
    BART_REALTIME_SUMMARY_MAX_LENGTH: int = int(os.getenv("BART_REALTIME_SUMMARY_MAX_LENGTH", 150))
    BART_REALTIME_SUMMARY_MIN_LENGTH: int = int(os.getenv("BART_REALTIME_SUMMARY_MIN_LENGTH", 50))
    
    # 最终会议纪要的 BART 模型生成参数
    BART_FINAL_SUMMARY_MAX_LENGTH: int = int(os.getenv("BART_FINAL_SUMMARY_MAX_LENGTH", 500))
    BART_FINAL_SUMMARY_MIN_LENGTH: int = int(os.getenv("BART_FINAL_SUMMARY_MIN_LENGTH", 100))

    # ==================== 数据库连接配置 ====================
    # MongoDB (文档数据库) 配置 - 主要用于会议元数据和原始发言存储
    MONGO_HOST: str = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT: int = int(os.getenv("MONGO_PORT", 27017))
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "meeting_db")
    MONGO_URI: str = os.getenv("MONGO_URI", f"mongodb://{MONGO_HOST}:{MONGO_PORT}") # 兼容性URI
    MONGO_LOG_TTL: int = int(os.getenv("MONGO_LOG_TTL", 60 * 60 * 24 * 7)) # 日志保留时间 TTL (秒)
    
    # MongoDB 转录集合名称
    MONGO_TRANSCRIPT_COLLECTION_NAME: str = os.getenv("MONGO_TRANSCRIPT_COLLECTION_NAME", "meeting_transcripts")

    # Milvus (向量数据库) 配置 - 通用部分
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER: str = os.getenv("MILVUS_USER", "root") # Milvus 用户名
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "Milvus") # Milvus 密码
    MILVUS_ALIAS: str = os.getenv("MILVUS_ALIAS", "default") # Milvus 连接别名
    MILVUS_METRIC_TYPE: str = os.getenv("MILVUS_METRIC_TYPE", "L2") # L2, IP, COSINE
    MILVUS_INDEX_TYPE: str = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT") # IVF_FLAT, HNSW
    MILVUS_NLIST: int = int(os.getenv("MILVUS_NLIST", 1024))
    MILVUS_NPROBE: int = int(os.getenv("MILVUS_NPROBE", 64))
    MILVUS_SHARDS_NUM: int = int(os.getenv("MILVUS_SHARDS_NUM", 2)) # Milvus 分片数量
    MILVUS_CONSISTENCY_LEVEL: str = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Bounded") # Strong, Bounded, Session, Eventually
    MILVUS_MAX_QUERY_LIMIT: int = int(os.getenv("MILVUS_MAX_QUERY_LIMIT", 16384)) # Milvus 查询结果最大数量
    MILVUS_DIMENSION: int = int(os.getenv("MILVUS_DIMENSION", 512)) # Milvus 向量维度 (BGE 模型维度)

    # --- Milvus 集合 1: 声纹特征 (voice_prints) ---
    MILVUS_VOICE_COLLECTION_NAME: str = os.getenv("MILVUS_VOICE_COLLECTION_NAME", "voice_prints")
    # 声纹相似度阈值，用于判断是否为同一人
    VOICEPRINT_SIMILARITY_THRESHOLD: float = float(os.getenv("VOICEPRINT_SIMILARITY_THRESHOLD", 0.8)) 
    
    @property
    def MILVUS_VOICE_SCHEMA_FIELDS(self) -> List[FieldSchema]:
        """声纹集合的 Schema 字段定义。"""
        return [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="user_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.MILVUS_DIMENSION)
        ]

    # --- Milvus 集合 2: 会议文本嵌入 (meeting_embeddings) ---
    MILVUS_MEETING_COLLECTION_NAME: str = os.getenv("MILVUS_MEETING_COLLECTION_NAME", "meeting_embeddings")
    
    @property
    def MILVUS_MEETING_SCHEMA_FIELDS(self) -> List[FieldSchema]:
        """会议文本嵌入集合的 Schema 字段定义。"""
        return [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), # 自动生成 ID
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.MILVUS_DIMENSION),
            FieldSchema(name="mongo_id", dtype=DataType.VARCHAR, max_length=256) # 关联 MongoDB 文档 ID
        ]

    # Milvus 索引参数属性 (通用，两个集合都用这个)
    @property
    def MILVUS_INDEX_PARAMS(self) -> Dict[str, Any]:
        """
        根据配置动态生成 Milvus 索引参数。
        """
        # 根据索引类型构建不同的参数字典
        if self.MILVUS_INDEX_TYPE == "IVF_FLAT":
            return {
                "metric_type": self.MILVUS_METRIC_TYPE,
                "index_type": self.MILVUS_INDEX_TYPE,
                "params": {"nlist": self.MILVUS_NLIST}
            }
        elif self.MILVUS_INDEX_TYPE == "HNSW":
            return {
                "metric_type": self.MILVUS_METRIC_TYPE,
                "index_type": self.MILVUS_INDEX_TYPE,
                "params": {"M": 16, "efConstruction": 200} 
            }
        else:
            logger.warning(f"不支持的 Milvus 索引类型: {self.MILVUS_INDEX_TYPE}。使用默认参数。")
            return {
                "metric_type": self.MILVUS_METRIC_TYPE,
                "index_type": self.MILVUS_INDEX_TYPE,
                "params": {}
            }

    # Neo4j (图谱数据库) 配置 - 主要用于用户关系和会议内容关系存储
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # ==================== 业务逻辑和实时处理参数 ====================
    TOPIC_DETECTION_INTERVAL: int = int(os.getenv("TOPIC_DETECTION_INTERVAL", 5)) # 每 N 条发言尝试检测一次话题
    TOPIC_DETECTION_CONTEXT_LINES: int = int(os.getenv("TOPIC_DETECTION_CONTEXT_LINES", 10)) # 用于话题检测的上下文发言行数

    REALTIME_SUMMARY_CONTEXT_LINES: int = int(os.getenv("REALTIME_SUMMARY_CONTEXT_LINES", 20)) # 实时摘要的上下文发言行数 
    SUMMARY_CONTEXT_LINES: int = int(os.getenv("SUMMARY_CONTEXT_LINES", 50)) # 最终会议纪要摘要的上下文发言行数 (用于 LLM 生成)
    
    # 问答功能使用的上下文发言行数
    QA_CONTEXT_LINES: int = int(os.getenv("QA_CONTEXT_LINES", 50)) 
    # 问答的 LLM 模型生成参数
    LLM_QA_MAX_LENGTH: int = int(os.getenv("LLM_QA_MAX_LENGTH", 200)) 

    # 行动项提取的 LLM 模型生成参数
    LLM_ACTION_ITEMS_MAX_LENGTH: int = int(os.getenv("LLM_ACTION_ITEMS_MAX_LENGTH", 250)) 
    LLM_ACTION_ITEMS_NUM_SEQUENCES: int = int(os.getenv("LLM_ACTION_ITEMS_NUM_SEQUENCES", 1)) 

    # ==================== 其他业务逻辑配置 ====================
    ROLE_WEIGHTS: dict = { # 角色权重，例如用于权限或优先级计算
        'host': 1.0,
        'client': 0.7,
        'member': 0.5,
        'member': 0.5,
        'guest': 0.3 
    }

    # ==================== 预定义实体 (用于简单的实体提取) ====================
    PREDEFINED_PERSONS: List[str] = ["李明", "王芳", "张三", "赵丽", "钱涛"]
    PREDEFINED_TOPICS: List[str] = [
        "线上推广方案", "短视频平台", "抖音", "快手", "预算", 
        "产品迭代", "用户反馈模块", "市场策略会议", "预算报告"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 确保所有模型目录存在
        # 这些目录现在都直接在 BASE_DIR / "models" 下
        # VAD 模型路径的父目录
        self.VAD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # STT 模型路径的父目录
        self.WHISPER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # ECAPA-TDNN 模型目录
        self.ECAPA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # LLM 模型路径的父目录
        self.LLM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # BGE 模型目录
        self.BGE_MODEL_PATH.mkdir(parents=True, exist_ok=True) 
        # Summary 模型路径的父目录
        self.SUMMARY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("所有模型目录已检查或创建。")

# 实例化设置对象
settings = Settings()

# 调试信息
logger.info(f"DEBUG Mode: {settings.DEBUG}")
logger.info(f"App Host: {settings.HOST}, Port: {settings.PORT}")
logger.info(f"Hugging Face Hub Offline Mode: {settings.HF_HUB_OFFLINE}")
# 隐藏 HF_TOKEN 的值，只显示是否已设置
logger.info(f"Hugging Face Token Set: {'Yes' if settings.HF_TOKEN else 'No'}") 
logger.info(f"MongoDB Host: {settings.MONGO_HOST}, Port: {settings.MONGO_PORT}, DB Name: {settings.MONGO_DB_NAME}")
logger.info(f"MongoDB Transcript Collection: {settings.MONGO_TRANSCRIPT_COLLECTION_NAME}") 
logger.info(f"Milvus Host: {settings.MILVUS_HOST}, Port: {settings.MILVUS_PORT}")
logger.info(f"Milvus User: {settings.MILVUS_USER}") 
logger.info(f"Milvus Password Set: {'Yes' if settings.MILVUS_PASSWORD else 'No'}") 
logger.info(f"Milvus Voice Collection: {settings.MILVUS_VOICE_COLLECTION_NAME}") 
logger.info(f"Milvus Meeting Collection: {settings.MILVUS_MEETING_COLLECTION_NAME}") 
logger.info(f"Milvus Dimension: {settings.MILVUS_DIMENSION}") 
logger.info(f"Milvus Index Params: {settings.MILVUS_INDEX_PARAMS}") 
logger.info(f"Neo4j URI: {settings.NEO4J_URI}")
logger.info(f"STT Model: {settings.WHISPER_MODEL_NAME}, Local Path: {settings.WHISPER_MODEL_PATH}") 
logger.info(f"Diarization Model (Hub): {settings.PYANNOTE_DIARIZATION_MODEL}") 
logger.info(f"Speaker Embedding Model (Hub): {settings.PYANNOTE_EMBEDDING_MODEL}, Local Dir: {settings.ECAPA_MODEL_DIR}") 
logger.info(f"LLM Model: {settings.LLM_MODEL_NAME}, Local Path: {settings.LLM_MODEL_PATH}") 
logger.info(f"BGE Model: {settings.BGE_MODEL_NAME}, Local Path: {settings.BGE_MODEL_PATH}") 
logger.info(f"Summary Model: {settings.SUMMARY_MODEL_HUB_NAME}, Local Path: {settings.SUMMARY_MODEL_PATH}") 
logger.info(f"Use CUDA: {settings.USE_CUDA}")

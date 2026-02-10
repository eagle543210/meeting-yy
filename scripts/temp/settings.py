# config/settings.py
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 语音识别和声纹相关的设置
    VOICE_SAMPLE_RATE: int = 16000 # 语音采样率
    VOICE_EMBEDDING_DIM: int = 192 
    
    # --- Milvus 配置 ---
    MILVUS_HOST: str = "localhost"  
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION_NAME: str = "voice_prints" 
    MILVUS_INDEX_PARAMS: dict = { 
        "index_type": "IVF_FLAT",
        "metric_type": "L2",        
        "params": {"nlist": 1024}
    }
    VOICE_COLLECTION: str = "voice_embeddings"
    MILVUS_USER: Optional[str] = "root" 
    MILVUS_PASSWORD: Optional[str] = "Milvus" 

    # --- MongoDB 配置 ---
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "meeting_system"
    MONGO_LOG_TTL: int = 2592000    # 日志保留30天（秒）
    
    # --- Neo4j 配置 ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "yy123123"
    KG_NODE_LABEL: str = "Entity"
    KG_NODE_PROPERTY: str = "name"
    KG_RELATION_TYPE: str = "RELATION"
    KG_RELATION_PROPERTY: str = "type"
    
    # 新增 STT 服务配置
    ENABLE_STT: bool = True # 语音转文本功能
      
    # --- BART 模型 ---
    MODEL_DIR: str = "models"    # 根模型存储目录

    # --- 新增摘要模型特定配置 ---
    SUMMARY_MODEL_HUB_NAME: str = "facebook/bart-large-cnn" # Hugging Face Hub 上的原始名称
    SUMMARY_MODEL_LOCAL_DIR_NAME: str = "bart-large-cnn"    # 本地模型文件夹的名称
    # --- 新增结束 ---
    
    # --- 声纹模型配置 (ECAPA) ---
    ECAPA_MODEL_DIR: str = "M:/meeting/models/ecapa_tdnn"    # 模型存储路径
    VOICE_SAMPLE_RATE: int = 16000    # 音频采样率
    VOICEPRINT_SIMILARITY_THRESHOLD: float = 35000.0
    MAX_VOICE_DURATION: int = 30    # 最大音频时长(秒)

    # --- 角色权重配置 ---
    DEFAULT_ROLE: str = "guest" # 角色默认值 
    ROLE_WEIGHTS: dict = {
        'host': 1.0,
        'client': 0.7,
        'member': 0.5
    }

    class Config:
        env_file = ".env" # 确保这个配置指向你的 .env 文件

settings = Settings()    # 单例配置对象
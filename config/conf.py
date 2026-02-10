# config.py

from typing import Optional

class AppConfig:
    """
    应用程序的配置类，包含数据库和模型路径等信息。
    """
    # --- Milvus 配置 ---
    MILVUS_HOST: str = "localhost"  
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION_NAME: str = "meeting_embeddings" 
    MILVUS_DIMENSION: int = 512 # BGE 模型输出的维度
    MILVUS_INDEX_PARAMS: dict = { 
        "index_type": "IVF_FLAT",
        "metric_type": "L2",       
        "params": {"nlist": 1024}
    }
    MILVUS_USER: Optional[str] = "root" 
    MILVUS_PASSWORD: Optional[str] = "Milvus" 

    # --- MongoDB 配置 ---
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "meeting_db"
    MONGO_COLLECTION_NAME: str = "meeting_segments" # 存储原始文本和元数据
    MONGO_LOG_TTL: int = 2592000 # 日志保留30天（秒），对于我们这里是文档过期时间

    # --- Neo4j 配置 (保留，但目前代码未直接使用) ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "yy123123"
    KG_NODE_LABEL: str = "Entity"
    KG_NODE_PROPERTY: str = "name"
    KG_RELATION_TYPE: str = "RELATION"
    KG_RELATION_PROPERTY: str = "type"

    # --- BAAI/bge-small-zh-v1.5 模型配置 ---
    BGE_MODEL_PATH: str = "M:/meeting/models/bge/small"
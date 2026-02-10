# init_milvus.py
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from config.settings import settings

def init_collection():
    # 1. 连接
    connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    
    # 2. 定义Schema（兼容v2.x）
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=192),
        FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=50)
    ]
    schema = CollectionSchema(fields, description="声纹特征库")
    
    # 3. 创建集合
    if not utility.has_collection(settings.VOICE_COLLECTION):
        collection = Collection(settings.VOICE_COLLECTION, schema)
        
        # 4. 必须创建索引（v2.x关键区别）
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        print(f"集合 {settings.VOICE_COLLECTION} 初始化完成（含索引）")
    else:
        print(f"集合 {settings.VOICE_COLLECTION} 已存在")

if __name__ == "__main__":
    init_collection()
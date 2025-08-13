# services/milvus_connector.py
from pymilvus import connections, Collection, utility
from config.settings import settings
from typing import List, Dict, Optional

class MilvusOperator:
    def __init__(self):
        """初始化Milvus连接"""
        connections.connect(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
    
    def create_collection(self, collection_name: str, dimension: int = 192) -> bool:
        """创建向量集合"""
        if utility.has_collection(collection_name):
            return False
            
        fields = [
            {"name": "id", "type": "VARCHAR", "is_primary": True},
            {"name": "embedding", "type": "FLOAT_VECTOR", "dim": dimension},
            {"name": "metadata", "type": "JSON"}
        ]
        schema = {
            "auto_id": False,
            "description": "Meeting system embeddings",
            "fields": fields
        }
        Collection(name=collection_name, schema=schema)
        return True

    def insert_embeddings(self, collection_name: str, data: List[Dict]) -> List[str]:
        """插入向量数据"""
        collection = Collection(collection_name)
        insert_result = collection.insert(data)
        collection.flush()
        return insert_result.primary_keys

    def search_similar(self, collection_name: str, vectors: List[List[float]], top_k: int = 5) -> List[List[Dict]]:
        """向量相似搜索"""
        collection = Collection(collection_name)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        results = collection.search(
            vectors, 
            "embedding",
            param=search_params,
            limit=top_k
        )
        return [
            [{"id": hit.id, "distance": hit.distance} for hit in result]
            for result in results
        ]

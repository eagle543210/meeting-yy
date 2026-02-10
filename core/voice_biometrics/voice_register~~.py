# M:\meeting\core\voice_biometrics\voice_register.py

import os
import logging
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pymilvus.exceptions import MilvusException
from config.settings import settings # 确保这个路径正确

logger = logging.getLogger(__name__)

class VoiceRegistration:
    def __init__(self):
        self.collection_name = getattr(settings, 'MILVUS_COLLECTION_NAME', "voice_embeddings")
        self.milvus_host = getattr(settings, 'MILVUS_HOST', "localhost")
        self.milvus_port = getattr(settings, 'MILVUS_PORT', "19530")
        self.embedding_dim = getattr(settings, 'VOICE_EMBEDDING_DIM', 192) # 确保这里的维度和 ECAPAWrapper 输出的维度一致

        logger.info(f"正在连接 Milvus 数据库: {self.milvus_host}:{self.milvus_port}")
        try:
            connections.connect(host=self.milvus_host, port=self.milvus_port)
            logger.info("Milvus 数据库连接成功。")
            self._initialize_collection()
        except Exception as e:
            logger.critical(f"❌ 无法连接或初始化 Milvus 数据库: {e}", exc_info=True)
            raise RuntimeError(f"Milvus 连接/初始化失败: {e}")

    def _initialize_collection(self):
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Milvus 集合 '{self.collection_name}' 已存在，正在加载。")
                self.collection = Collection(self.collection_name)
                # 检查 schema 是否符合预期
                if not self._check_collection_schema(self.collection):
                    logger.warning(f"现有集合 '{self.collection_name}' 的 schema 不符合预期。建议清空或重建。")
                    # 你可能需要在此处添加逻辑来处理 schema 不匹配的情况，例如删除并重建
                    # 或者跳过重建，并确保你的查询只使用存在的字段
            else:
                logger.info(f"Milvus 集合 '{self.collection_name}' 不存在，正在创建。")
                self._create_collection()
            
            # 确保 collection 加载到内存中才能进行搜索
            self.collection.load()
            logger.info(f"Milvus 集合 '{self.collection_name}' 已加载到内存。")

        except Exception as e:
            logger.critical(f"❌ Milvus 集合初始化失败: {e}", exc_info=True)
            raise RuntimeError(f"Milvus 集合初始化失败: {e}")

    def _check_collection_schema(self, collection: Collection) -> bool:
        """检查现有 collection 的 schema 是否符合预期"""
        fields = {f.name: f for f in collection.schema.fields}
        
        # 检查 'id' 字段
        if 'id' not in fields or fields['id'].dtype not in [DataType.VARCHAR, DataType.INT64] or not fields['id'].is_primary:
            logger.warning("Milvus collection schema 缺少 primary key 'id' 或类型不匹配。")
            return False
        
        # 检查 'embedding' 字段
        if 'embedding' not in fields or fields['embedding'].dtype != DataType.FLOAT_VECTOR or fields['embedding'].dim != self.embedding_dim:
            logger.warning("Milvus collection schema 缺少 'embedding' 向量字段或维度不匹配。")
            return False
            
        # 检查 'role' 字段
        if 'role' not in fields or fields['role'].dtype != DataType.VARCHAR:
            logger.warning("Milvus collection schema 缺少 'role' 字符串字段或类型不匹配。")
            return False
            
        return True

    def _create_collection(self):
        # 定义字段 Schema
        fields = [
            # 关键修改 1: 定义 'id' 字段作为 primary_key，类型为 VARCHAR
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256), 
            # 关键修改 2: 定义 'embedding' 向量字段，维度与 ECAPAWrapper 保持一致
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            # 关键修改 3: 定义 'role' 字段为 VARCHAR
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=64) 
        ]

        # 定义 Collection Schema
        schema = CollectionSchema(
            fields,
            description="Voiceprint embeddings for speaker recognition"
        )

        # 创建 Collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default', # 默认的 Milvus 连接
            shards_num=2
        )
        logger.info(f"Milvus 集合 '{self.collection_name}' 创建成功。")

        # 为向量字段创建索引
        index_params = {
            "metric_type":"L2", # 使用 L2 距离，与搜索查询一致
            "index_type":"IVF_FLAT",
            "params":{"nlist":1024}
        }
        self.collection.create_index(
            field_name="embedding", 
            index_params=index_params
        )
        logger.info(f"已为集合 '{self.collection_name}' 的 'embedding' 字段创建索引。")

    def close(self):
        try:
            if hasattr(self, 'collection') and self.collection:
                self.collection.release() # 释放集合，从内存中卸载
                logger.info(f"Milvus 集合 '{self.collection_name}' 已从内存中释放。")
            # connections.disconnect() # 通常不需要在每次 VoiceRegistration 实例关闭时断开全局连接
        except Exception as e:
            logger.error(f"关闭 Milvus 连接时发生错误: {str(e)}", exc_info=True)
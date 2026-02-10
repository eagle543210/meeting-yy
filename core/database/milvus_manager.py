# M:\meeting\core\database\milvus_manager.py

import logging
import asyncio
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pymilvus.exceptions import MilvusException

logger = logging.getLogger(__name__)

class MilvusManager:
    """
    管理与 Milvus 向量数据库的连接和操作。
    用于存储和检索声纹嵌入向量。
    """
    def __init__(self, settings_obj: Any):
        self.settings = settings_obj
        self.collection_name = self.settings.MILVUS_COLLECTION_NAME
        self.host = self.settings.MILVUS_HOST
        self.port = self.settings.MILVUS_PORT
        self.embedding_dim = self.settings.VOICE_EMBEDDING_DIM
        self.index_params = self.settings.MILVUS_INDEX_PARAMS
        self.collection: Optional[Collection] = None
        self._connected: bool = False
        logger.info(f"MilvusManager 初始化完成。连接到 {self.host}:{self.port}, 集合: {self.collection_name}")

    async def connect(self):
        """
        异步连接到 Milvus 数据库，并确保集合存在且已加载。
        """
        if self._connected:
            logger.info("Milvus 已连接，跳过重复连接。")
            return

        logger.info(f"正在连接到 Milvus: {self.host}:{self.port}...")
        try:
            # 异步连接 Milvus
            await asyncio.to_thread(
                connections.connect,
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info("✅ Milvus 连接成功。")

            # 检查集合是否存在，如果不存在则创建
            if not await asyncio.to_thread(utility.has_collection, self.collection_name):
                logger.info(f"集合 '{self.collection_name}' 不存在，正在创建...")
                await self._create_collection()
                logger.info(f"集合 '{self.collection_name}' 创建成功。")
            else:
                logger.info(f"集合 '{self.collection_name}' 已存在。")

            # 获取集合并加载到内存
            self.collection = await asyncio.to_thread(Collection, self.collection_name)
            await asyncio.to_thread(self.collection.load)
            self._connected = True
            logger.info(f"✅ 集合 '{self.collection_name}' 已加载到内存。")

        except MilvusException as e:
            logger.critical(f"❌ Milvus 连接或集合操作失败: {e}", exc_info=True)
            self._connected = False
            raise RuntimeError(f"Milvus 初始化失败: {str(e)}")
        except Exception as e:
            logger.critical(f"❌ 未知错误导致 Milvus 连接或集合操作失败: {e}", exc_info=True)
            self._connected = False
            raise RuntimeError(f"Milvus 初始化失败: {str(e)}")

    def is_connected(self) -> bool:
        """
        检查 Milvus 连接是否活跃。
        """
        return self._connected and self.collection is not None

    async def _create_collection(self):
        """
        创建 Milvus 集合，定义字段和索引。
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="username", dtype=DataType.VARCHAR, max_length=256), # <--- 添加 username 字段
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=256),     # <--- 添加 role 字段
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        schema = CollectionSchema(fields, description="Voice print embeddings")
        
        # 异步创建集合
        self.collection = await asyncio.to_thread(Collection, self.collection_name, schema)

        # 异步创建索引
        await asyncio.to_thread(
            self.collection.create_index,
            field_name="embedding",
            index_params=self.index_params
        )
        logger.info(f"索引已为集合 '{self.collection_name}' 的 'embedding' 字段创建。")

    async def insert_voice_print(self, user_id: str, embedding: List[float], role: str, username: str):
        """
        向 Milvus 集合插入声纹数据。
        Args:
            user_id (str): 用户的唯一ID。
            embedding (List[float]): 声纹嵌入向量。
            role (str): 用户的角色。
            username (str): 用户的名称。
        """
        if not self.is_connected():
            raise RuntimeError("Milvus 未连接，无法插入声纹。")

        data = [
            [str(user_id)],  # id 字段
            [user_id],       # user_id 字段
            [username],      # username 字段
            [role],          # role 字段
            [embedding]      # embedding 字段
        ]
        try:
            # 异步插入数据
            await asyncio.to_thread(self.collection.insert, data)
            await asyncio.to_thread(self.collection.flush) # 刷新数据到磁盘
            logger.info(f"声纹数据 for user_id '{user_id}' 已成功插入 Milvus。")
        except MilvusException as e:
            logger.error(f"插入声纹数据失败 for user_id '{user_id}': {e}", exc_info=True)
            raise RuntimeError(f"Milvus 插入操作失败: {str(e)}")

    async def search_voice_prints(self, query_embedding: List[float], top_k: int = 1) -> List[Dict[str, Any]]:
        """
        在 Milvus 集合中搜索相似的声纹。
        Args:
            query_embedding (List[float]): 要查询的声纹向量。
            top_k (int): 返回最相似结果的数量。
        Returns:
            List[Dict[str, Any]]: 搜索结果列表，每个结果包含 'user_id', 'username', 'role' 和 'distance'。
        """
        if not self.is_connected():
            raise RuntimeError("Milvus 未连接，无法搜索声纹。")

        search_params = {
            "metric_type": self.index_params["metric_type"],
            "params": {"nprobe": 10} # 根据索引类型调整 nprobe
        }

        try:
            # 异步搜索
            results = await asyncio.to_thread(
                self.collection.search,
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["user_id", "username", "role"] # <--- 确保这里包含 username 和 role
            )

            formatted_results = []
            for hit in results[0]: # results[0] 包含查询结果
                formatted_results.append({
                    "user_id": hit.entity.get("user_id"),
                    "username": hit.entity.get("username"),
                    "role": hit.entity.get("role"),
                    "distance": hit.distance
                })
            logger.debug(f"Milvus 搜索完成，返回 {len(formatted_results)} 个结果。")
            return formatted_results
        except MilvusException as e:
            logger.error(f"搜索声纹数据失败: {e}", exc_info=True)
            raise RuntimeError(f"Milvus 搜索操作失败: {str(e)}")

    async def load_all_voice_prints(self) -> Dict[str, Dict[str, Any]]:
        """
        从 Milvus 集合加载所有声纹的元数据。
        Returns:
            Dict[str, Dict[str, Any]]: 以 user_id 为键的声纹元数据字典。
        """
        if not self.is_connected():
            raise RuntimeError("Milvus 未连接，无法加载所有声纹。")
        
        try:
            # 使用 query 方法获取所有数据，并指定输出字段
            # 确保 Milvus 集合中存在这些字段
            results = await asyncio.to_thread(
                self.collection.query,
                expr="id like '%%'", # 查询所有数据
                output_fields=["id", "user_id", "username", "role", "embedding"] # <--- 确保这里包含 username 和 role
            )

            all_voice_prints = {}
            for entity in results:
                # Milvus query 返回的 entity 是字典
                all_voice_prints[entity["user_id"]] = {
                    "id": entity["id"],
                    "user_id": entity["user_id"],
                    "username": entity["username"],
                    "role": entity["role"],
                    "embedding": entity["embedding"]
                }
            logger.info(f"成功从 Milvus 加载 {len(all_voice_prints)} 条声纹元数据。")
            return all_voice_prints
        except MilvusException as e:
            logger.error(f"从 Milvus 加载所有声纹元数据失败: {e}", exc_info=True)
            raise RuntimeError(f"Milvus 加载所有声纹操作失败: {str(e)}")

    async def close(self):
        """
        关闭 Milvus 连接，释放集合。
        """
        logger.info("正在关闭 Milvus 连接...")
        if self.collection:
            try:
                await asyncio.to_thread(self.collection.release)
                logger.info(f"集合 '{self.collection_name}' 已从内存中释放。")
            except MilvusException as e:
                logger.warning(f"释放 Milvus 集合 '{self.collection_name}' 失败: {e}")
            self.collection = None
        
        try:
            # connections.disconnect("default") # Milvus 2.x 推荐不显式断开，让连接池管理
            logger.info("Milvus 连接已关闭。")
        except Exception as e:
            logger.warning(f"关闭 Milvus 连接时发生错误: {e}")
        self._connected = False


# services/milvus_service.py
import logging 
import asyncio 
from typing import List, Dict, Any, Optional 
from pymilvus import ( 
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, 
    MilvusException, 
    SearchResult, 
    Hit
)

# 从项目配置中导入设置
from config.settings import settings 

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

class MilvusManager:
    """
    MilvusManager 负责管理与 Milvus 向量数据库的连接、集合的创建、数据的插入和搜索。
    它被设计为通用型，可以通过传入不同的集合名称和 Schema 来管理多个集合。
    """
    def __init__(self, config: settings, collection_name: str, schema_fields: List[FieldSchema]):
        """
        初始化 MilvusManager 实例。
        Args:
            config (settings): 应用程序的配置设置对象。
            collection_name (str): 此 MilvusManager 实例将管理的集合名称。
            schema_fields (List[FieldSchema]): 此集合的 Schema 字段定义。
        """
        self.settings = config # 存储配置设置
        self.collection_name = collection_name # 存储当前实例管理的集合名称
        self.schema_fields = schema_fields # 存储当前实例管理的集合 Schema 字段
        self.collection: Optional[Collection] = None # Milvus 集合对象，初始化为 None
        self.is_connected = False # 连接状态标记
        
        logger.info(f"MilvusManager 初始化，目标: {self.settings.MILVUS_HOST}:{self.settings.MILVUS_PORT}, 集合: {self.collection_name}, 维度: {self.settings.MILVUS_DIMENSION}")

    async def connect_with_retry(self, overwrite_collection: bool = False, retries: int = 5, delay: int = 5):
        """
        异步连接到 Milvus，并在失败时重试。
        """
        for i in range(retries):
            try:
                await self.connect(overwrite_collection)
                return
            except MilvusException as e:
                if i < retries - 1:
                    logger.warning(f"Milvus 连接失败，将在 {delay} 秒后重试... ({i+1}/{retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Milvus 连接在多次重试后仍然失败。")
                    raise e

    async def connect(self, overwrite_collection: bool = False):
        """
        异步连接到 Milvus 数据库并验证/创建集合。
        此方法会在启动时被调用，以确保 Milvus 服务准备就绪。
        Args:
            overwrite_collection (bool): 如果为 True，则在集合存在时删除并重新创建。
        """
        if self.is_connected and self.collection: # 如果已经连接且集合已加载，则直接返回
            logger.info(f"Milvus 集合 '{self.collection_name}' 已连接并加载。")
            return

        try:
            # 建立与 Milvus 的连接 (通用连接，只连接一次)
            # 检查是否已经有默认连接，避免重复连接
            if not connections.has_connection(self.settings.MILVUS_ALIAS):
                logger.info(f"尝试连接 Milvus: {self.settings.MILVUS_HOST}:{self.settings.MILVUS_PORT}...")
                connections.connect(
                    alias=self.settings.MILVUS_ALIAS, # 使用配置中的别名
                    host=self.settings.MILVUS_HOST, # Milvus 主机地址
                    port=self.settings.MILVUS_PORT, # Milvus 端口
                    user=self.settings.MILVUS_USER, # Milvus 用户名
                    password=self.settings.MILVUS_PASSWORD, # Milvus 密码
                    secure=False # 是否使用安全连接（HTTPS/SSL），根据 Milvus 配置设置
                )
                logger.info("Milvus 连接成功。")
            else:
                logger.info("Milvus 连接已存在，跳过重新连接。")

            await self._create_collection_if_not_exists(overwrite_collection) # 连接成功后，检查并创建集合
            self.is_connected = True # 设置连接状态为 True
            logger.info(f"MilvusManager for '{self.collection_name}' 连接和集合检查/创建完成。")
        except MilvusException as e:
            logger.error(f"连接或操作 Milvus 失败 (集合: {self.collection_name}): {e}", exc_info=True) # 记录 Milvus 特定异常
            self.is_connected = False
            self.collection = None
            raise # 重新抛出异常，让调用方处理
        except Exception as e:
            logger.error(f"MilvusManager 初始化失败 (集合: {self.collection_name}): {e}", exc_info=True) # 记录其他通用异常
            self.is_connected = False
            self.collection = None
            raise # 重新抛出异常，让调用方处理

    async def _create_collection_if_not_exists(self, overwrite_collection: bool):
        """
        异步创建 Milvus 集合（如果不存在），或根据需要覆盖现有集合。
        定义集合的字段、主键和向量维度，并创建索引。
        Args:
            overwrite_collection (bool): 如果为 True，则在集合存在时删除并重新创建。
        """
        try:
            has_collection = await asyncio.to_thread(utility.has_collection, self.collection_name)

            if has_collection:
                if overwrite_collection:
                    logger.warning(f"Milvus 集合 '{self.collection_name}' 已存在。根据请求正在覆盖。")
                    await asyncio.to_thread(utility.drop_collection, self.collection_name)
                    self.collection = None # 确保旧的 collection 引用被清除
                    has_collection = False # 标记为已删除，以便重新创建
                else:
                    logger.info(f"Milvus 集合 '{self.collection_name}' 已存在。正在加载现有集合。")
                    self.collection = Collection(self.collection_name)
                    # 验证现有集合的 Schema 是否与预期匹配
                    current_fields = {f.name: f for f in self.collection.schema.fields}
                    schema_mismatch = False
                    for expected_field in self.schema_fields:
                        if expected_field.name not in current_fields:
                            logger.critical(f"集合 '{self.collection_name}' 缺少预期字段: '{expected_field.name}'。")
                            schema_mismatch = True
                            break
                        # 可以在这里添加更多类型或维度检查
                    
                    if schema_mismatch:
                        logger.critical(f"集合 '{self.collection_name}' 的 Schema 不匹配预期。请考虑设置 overwrite_collection=True 来重建集合。")
                        # 如果 Schema 不匹配，强制删除并重建，以确保 Schema 正确
                        logger.warning(f"强制删除并重建 Milvus 集合 '{self.collection_name}'，因为它 Schema 不匹配。")
                        await asyncio.to_thread(utility.drop_collection, self.collection_name)
                        self.collection = None
                        has_collection = False # 标记为已删除，以便重新创建
                    else:
                        # 检查索引是否存在，如果不存在则创建
                        if not await asyncio.to_thread(self.collection.has_index):
                            logger.info(f"集合 '{self.collection_name}' 存在但缺少索引，正在创建索引...")
                            await asyncio.to_thread(self.collection.create_index, 
                                                     field_name="embedding", # 在 'embedding' 字段上创建索引
                                                     index_params=self.settings.MILVUS_INDEX_PARAMS) # 索引参数
                            logger.info(f"索引创建完成，参数: {self.settings.MILVUS_INDEX_PARAMS}")
                        else:
                            logger.info(f"集合 '{self.collection_name}' 已存在且索引就绪。")
                        
                        logger.info(f"Milvus 集合 '{self.collection_name}' 尝试加载到内存。")
                        await asyncio.to_thread(self.collection.load) # 异步加载集合
                        logger.info(f"Milvus 集合 '{self.collection_name}' 已加载到内存。")
                        return # 如果不覆盖且已存在，则直接返回

            if not has_collection: # 如果集合不存在或者被覆盖了，则创建新集合
                logger.info(f"Milvus 集合 '{self.collection_name}' 不存在，正在创建。")
                schema = CollectionSchema(self.schema_fields, f"Embeddings for {self.collection_name}")
                self.collection = Collection(self.collection_name, schema) # 创建集合
                logger.info(f"Milvus 集合 '{self.collection_name}' 创建成功。")

                # 在集合创建后创建索引
                logger.info(f"正在为集合 '{self.collection_name}' 创建索引...")
                await asyncio.to_thread(self.collection.create_index, 
                                         field_name="embedding", # 在 'embedding' 字段上创建索引
                                         index_params=self.settings.MILVUS_INDEX_PARAMS) # 索引参数
                logger.info(f"索引创建完成，参数: {self.settings.MILVUS_INDEX_PARAMS}")

                # 索引创建后将集合加载到内存
                logger.info(f"Milvus 集合 '{self.collection_name}' 尝试加载到内存。")
                await asyncio.to_thread(self.collection.load) # 异步加载集合
                logger.info(f"Milvus 集合 '{self.collection_name}' 已加载到内存。")
            
        except MilvusException as e:
            logger.error(f"创建或加载 Milvus 集合失败 (集合: {self.collection_name}): {e}", exc_info=True)
            self.collection = None
            raise # 重新抛出异常
        except Exception as e:
            logger.error(f"Milvus 集合操作失败 (集合: {self.collection_name}): {e}", exc_info=True)
            self.collection = None
            raise # 重新抛出异常

    async def insert_data(self, data: List[Dict[str, Any]]) -> List[Any]:
        """
        异步向 Milvus 集合插入数据。
        Args:
            data (List[Dict[str, Any]]): 包含要插入的实体数据的字典列表。
                                         每个字典的键必须与集合 Schema 中定义的字段名称匹配。
        Returns:
            List[Any]: 插入成功后的主键 ID 列表。
        """
        if not self.is_connected or not self.collection:
            logger.error(f"Milvus 集合 '{self.collection_name}' 未连接或未就绪，无法插入数据。")
            return []
        
        try:
            # 'insert' 是一个阻塞调用，因此使用 asyncio.to_thread 在单独的线程中执行
            mutations = await asyncio.to_thread(self.collection.insert, data)
            # 确保数据立即刷新到磁盘，以便可以立即进行搜索
            await asyncio.to_thread(self.collection.flush)
            pks = mutations.primary_keys # 获取插入数据的主键 ID
            logger.debug(f"成功插入 {len(pks)} 条数据到集合 '{self.collection_name}'。")
            return pks
        except MilvusException as e:
            logger.error(f"插入数据到 Milvus 集合 '{self.collection_name}' 失败: {e}", exc_info=True)
            raise # 重新抛出异常
        except Exception as e:
            logger.error(f"插入数据到 Milvus 集合 '{self.collection_name}' 失败 (通用错误): {e}", exc_info=True)
            raise # 重新抛出异常

    async def search_data(self,
                          query_vectors: List[List[float]],
                          top_k: int = 1,
                          expr: Optional[str] = None,
                          output_fields: Optional[List[str]] = None) -> SearchResult:
        """
        异步在 Milvus 集合中搜索最相似的嵌入。
        Args:
            query_vectors (List[List[float]]): 要搜索的嵌入向量列表。
            top_k (int): 返回最相似结果的数量。
            expr (Optional[str]): Milvus 过滤表达式。
            output_fields (Optional[List[str]]): 要返回的字段列表。
        Returns:
            SearchResult: 包含相似结果的 pymilvus.SearchResult 对象。
        """
        if not self.is_connected or not self.collection:
            logger.error(f"Milvus 集合 '{self.collection_name}' 未连接或未就绪，无法搜索数据。")
            # 关键修复：返回一个空的 SearchResult 对象，而不是一个空列表
            return SearchResult()

        search_params = {
            "data": query_vectors,
            "anns_field": "embedding",
            "param": {"metric_type": self.settings.MILVUS_INDEX_PARAMS["metric_type"], "params": {"nprobe": self.settings.MILVUS_NPROBE}},
            "limit": top_k,
            "expr": expr,
            "output_fields": output_fields if output_fields else []
        }

        try:
            # 'search' 是一个阻塞调用，因此使用 asyncio.to_thread 在单独的线程中执行
            # 直接返回 pymilvus.SearchResult 对象，不再进行手动解析
            results: SearchResult = await asyncio.to_thread(self.collection.search, **search_params)
            # 打印调试日志，确保返回结果的类型正确
            if isinstance(results, SearchResult) and len(results) > 0:
                 logger.debug(f"Milvus 搜索完成，在集合 '{self.collection_name}' 中找到 {len(results[0])} 个结果。")
            else:
                 logger.debug(f"Milvus 搜索返回了非预期的结果类型或空结果。")
            return results
        except MilvusException as e:
            logger.error(f"从 Milvus 集合 '{self.collection_name}' 搜索数据失败: {e}", exc_info=True)
            return SearchResult()
        except Exception as e:
            logger.error(f"搜索数据失败 (集合: {self.collection_name}, 通用错误): {e}", exc_info=True)
            return SearchResult()

    async def get_data_by_ids(self, ids: List[Any], output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据 ID 列表从 Milvus 集合获取数据。
        Args:
            ids (List[Any]): 要查询的 ID 列表。
            output_fields (Optional[List[str]]): 要返回的字段列表。
        Returns:
            List[Dict[str, Any]]: 包含匹配数据的列表。
        """
        if not self.is_connected or not self.collection:
            logger.error(f"Milvus 集合 '{self.collection_name}' 未连接或未就绪，无法按 ID 获取数据。")
            return []
        
        if not ids:
            return []

        # 动态构建查询表达式，根据主键类型
        primary_key_name = None
        primary_key_dtype = None
        for field_schema in self.schema_fields:
            if field_schema.is_primary:
                primary_key_name = field_schema.name
                primary_key_dtype = field_schema.dtype
                break

        if not primary_key_name:
            logger.error(f"Milvus 集合 '{self.collection_name}' 未找到主键字段，无法按 ID 查询。")
            return []

        if primary_key_dtype == DataType.VARCHAR:
            expr_ids = ", ".join([f"'{_id}'" for _id in ids])
        elif primary_key_dtype == DataType.INT64:
            expr_ids = ", ".join([str(_id) for _id in ids])
        else:
            logger.error(f"不支持的 Milvus 主键类型 '{primary_key_dtype}' 进行 ID 查询。")
            return []

        expr = f"{primary_key_name} in [{expr_ids}]"
        
        try:
            # 'query' 是一个阻塞调用，因此使用 asyncio.to_thread 在单独的线程中执行
            results = await asyncio.to_thread(self.collection.query, expr=expr, output_fields=output_fields if output_fields else [], limit=len(ids))
            logger.debug(f"从集合 '{self.collection_name}' 查询到 {len(results)} 条数据。")
            return results
        except MilvusException as e:
            logger.error(f"从 Milvus 集合 '{self.collection_name}' 查询数据失败: {e}", exc_info=True)
            raise # 重新抛出异常
        except Exception as e:
            logger.error(f"按 ID 获取数据失败 (集合: {self.collection_name}, 通用错误): {e}", exc_info=True)
            raise # 重新抛出异常

    async def get_all_data(self, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        【已修复】
        异步获取 Milvus 集合中的所有数据。
        Args:
            output_fields (Optional[List[str]]): 要返回的字段列表。
        Returns:
            List[Dict[str, Any]]: 包含所有数据的列表。
        """
        if not self.is_connected or not self.collection:
            logger.error(f"Milvus 集合 '{self.collection_name}' 未连接或未就绪，无法获取所有数据。")
            return []
        
        try:
            
            primary_key_name = None
            for field_schema in self.schema_fields:
                if field_schema.is_primary:
                    primary_key_name = field_schema.name
                    break
            
            
            expr_to_use = f"{primary_key_name} != 'NEVER_MATCH_THIS_STRING'" if primary_key_name else "1 == 1"

            logger.info(f"正在从 Milvus 集合 '{self.collection_name}' 查询所有数据，使用表达式: '{expr_to_use}'")

            results = await asyncio.to_thread(self.collection.query, 
                                                 expr=expr_to_use, 
                                                 output_fields=output_fields if output_fields else [], 
                                                 limit=self.settings.MILVUS_MAX_QUERY_LIMIT)
            
            logger.debug(f"从集合 '{self.collection_name}' 查询到 {len(results)} 条数据。")
            return results
        except MilvusException as e:
            logger.error(f"从 Milvus 集合 '{self.collection_name}' 查询所有数据失败: {e}", exc_info=True)
            raise # 重新抛出异常
        except Exception as e:
            logger.error(f"获取所有数据失败 (集合: {self.collection_name}, 通用错误): {e}", exc_info=True)
            raise # 重新抛出异常

    async def query_data(self, expr: str, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据条件表达式查询 Milvus 集合中的数据。
        Args:
            expr (str): 查询表达式，例如 "speaker_id == 'user_1'"。
            output_fields (Optional[List[str]]): 要返回的字段列表。
        Returns:
            List[Dict[str, Any]]: 包含匹配数据的列表。
        """
        if not self.is_connected or not self.collection:
            logger.error(f"Milvus 集合 '{self.collection_name}' 未连接或未就绪，无法执行查询。")
            return []

        try:
            logger.info(f"正在从 Milvus 集合 '{self.collection_name}' 查询数据，表达式: '{expr}'")
            # 'query' 是一个阻塞调用，因此使用 asyncio.to_thread 在单独的线程中执行
            results = await asyncio.to_thread(self.collection.query, 
                                                 expr=expr, 
                                                 output_fields=output_fields if output_fields else [], 
                                                 limit=self.settings.MILVUS_MAX_QUERY_LIMIT)
            
            logger.debug(f"从集合 '{self.collection_name}' 查询到 {len(results)} 条数据。")
            return results
        except MilvusException as e:
            logger.error(f"从 Milvus 集合 '{self.collection_name}' 查询数据失败: {e}", exc_info=True)
            raise # 重新抛出异常
        except Exception as e:
            logger.error(f"查询数据失败 (集合: {self.collection_name}, 通用错误): {e}", exc_info=True)
            raise # 重新抛出异常

    async def close(self):
        """
        异步关闭 Milvus 连接并释放集合。
        注意：connections.disconnect() 会断开所有使用该 alias 的连接。
        """
        logger.info(f"正在关闭 MilvusManager for '{self.collection_name}'...")
        try:
            if self.collection: # 如果集合对象存在
                try:
                    logger.info(f"正在从内存中释放 Milvus 集合 '{self.collection_name}'。")
                    await asyncio.to_thread(self.collection.release) # 异步释放集合
                    logger.info(f"Milvus 集合 '{self.collection_name}' 已释放。")
                except MilvusException as e:
                    if "ConnectionNotExistException" in str(e) or "connection not exist" in str(e).lower() or "collection not loaded" in str(e).lower():
                        logger.warning(f"尝试释放 Milvus 集合 '{self.collection_name}' 时连接已不存在或集合未加载：{e}")
                    else:
                        logger.error(f"释放 Milvus 集合 '{self.collection_name}' 失败: {e}", exc_info=True)
                finally:
                    self.collection = None # 无论释放是否成功，都清空集合对象
            
            self.is_connected = False # 设置连接状态为 False
            logger.info(f"MilvusManager for '{self.collection_name}' 已关闭。")
        except Exception as e:
            logger.error(f"关闭 MilvusManager for '{self.collection_name}' 时发生通用错误: {e}", exc_info=True)

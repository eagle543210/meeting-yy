import logging
import time
from typing import List
import os
import sys

# 确保能正确导入 config.settings
# 假设脚本在项目根目录，或者在 M:\meeting\ 下
# 如果运行环境路径不对，需要调整 sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.settings import settings
except ImportError:
    print("错误：无法导入 config/settings.py。请确保您的文件路径和虚拟环境设置正确。")
    print(f"当前 sys.path: {sys.path}")
    sys.exit(1)

from pymilvus import connections, Collection, FieldSchema, DataType, utility, MilvusException, CollectionSchema

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MilvusTestManager:
    def __init__(self):
        self.collection_name = getattr(settings, 'MILVUS_COLLECTION_NAME', "voice_prints")
        self.host = getattr(settings, 'MILVUS_HOST', "localhost")
        self.port = getattr(settings, 'MILVUS_PORT', "19530")
        self.alias = getattr(settings, 'MILVUS_ALIAS', "default")
        self.embedding_dim = getattr(settings, 'VOICE_EMBEDDING_DIM', 192) # 确保与实际声纹维度一致
        self.metric_type = getattr(settings, 'MILVUS_METRIC_TYPE', "L2")
        self.index_type = getattr(settings, 'MILVUS_INDEX_TYPE', "IVF_FLAT")
        self.nlist = getattr(settings, 'MILVUS_NLIST', 1024)
        self.nprobe = getattr(settings, 'MILVUS_NPROBE', 64)

        self.collection: Optional[Collection] = None

    def connect(self):
        """连接到 Milvus 服务。"""
        try:
            logger.info(f"正在连接到 Milvus: {self.host}:{self.port}, alias: {self.alias}")
            if connections.has_connection(self.alias):
                connections.remove_connection(self.alias)
                logger.info(f"移除了别名 '{self.alias}' 的旧 Milvus 连接。")
            connections.connect(alias=self.alias, host=self.host, port=self.port)
            logger.info("✅ 已连接到 Milvus 服务。")
        except MilvusException as me:
            logger.critical(f"❌ Milvus 连接失败: {me}", exc_info=True)
            raise RuntimeError(f"Milvus 连接失败: {str(me)}")
        except Exception as e:
            logger.critical(f"❌ 连接 Milvus 发生意外错误: {e}", exc_info=True)
            raise RuntimeError(f"无法连接 Milvus: {str(e)}")

    def drop_and_create_collection(self):
        """强制删除并重新创建 Milvus 集合。"""
        self.connect() # 确保已连接

        try:
            if utility.has_collection(self.collection_name, using=self.alias):
                logger.warning(f"集合 '{self.collection_name}' 已存在，正在删除旧集合...")
                utility.drop_collection(self.collection_name, using=self.alias)
                logger.info(f"✅ 旧集合 '{self.collection_name}' 已成功删除。")
            else:
                logger.info(f"集合 '{self.collection_name}' 不存在，准备创建新集合。")

            # 定义集合 Schema (与您的 milvus_manager.py 保持一致)
            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256, description="主键/用户ID"),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim, description="声纹特征向量"),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=256, description="实际用户ID (与pk一致)"),
                FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=128, description="用户角色"),
                FieldSchema(name="timestamp", dtype=DataType.INT64, description="插入时间戳 (毫秒)")
            ]
            schema = CollectionSchema(fields, description="Voice print embeddings collection")

            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self.alias,
                shards_num=getattr(settings, 'MILVUS_SHARDS_NUM', 2),
                consistency_level=getattr(settings, 'MILVUS_CONSISTENCY_LEVEL', "Bounded")
            )
            logger.info(f"✅ 新集合 '{self.collection_name}' 创建成功。")
            logger.info(f"实际创建的 Milvus 集合 Schema: {self.collection.schema}") # 打印实际创建的 Schema

            # 创建索引
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": self.nlist}
            }
            logger.info(f"正在为集合 '{self.collection_name}' 创建索引，参数: {index_params}")
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.info(f"✅ 集合 '{self.collection_name}' 索引创建成功。")

            # 加载集合到内存
            self.collection.load()
            logger.info(f"集合 '{self.collection_name}' 已加载到内存。")
            logger.info(f"✅ Milvus 集合 '{self.collection_name}' 已完全就绪。")

        except MilvusException as me:
            logger.critical(f"❌ Milvus 集合创建或初始化失败: {me}", exc_info=True)
            raise RuntimeError(f"Milvus 集合创建或初始化失败: {str(me)}")
        except Exception as e:
            logger.critical(f"❌ 初始化集合发生意外错误: {e}", exc_info=True)
            raise RuntimeError(f"无法创建或初始化 Milvus 集合: {str(e)}")

    def insert_test_data(self, test_pk: str, test_embedding: List[float], test_role: str):
        """插入一条测试数据。"""
        if self.collection is None:
            raise RuntimeError("集合未初始化，无法插入数据。")

        try:
            entities_to_insert = [
                {
                    "pk": test_pk,
                    "embedding": test_embedding,
                    "user_id": test_pk,
                    "role": test_role,
                    "timestamp": int(time.time() * 1000)
                }
            ]
            logger.info(f"正在插入测试数据，pk='{test_pk}'...")
            mr = self.collection.insert(entities_to_insert)
            self.collection.flush() # 立即刷新，确保数据可见
            logger.info(f"✅ 成功插入测试数据，插入ID: {mr.primary_keys}")
        except MilvusException as me:
            logger.error(f"❌ Milvus 插入测试数据失败: {me}", exc_info=True)
            raise RuntimeError(f"Milvus 插入测试数据失败: {str(me)}")

    def query_test_data(self, test_pk: str):
        """查询插入的测试数据，确认 pk 字段可用。"""
        if self.collection is None:
            raise RuntimeError("集合未初始化，无法查询数据。")

        try:
            logger.info(f"正在查询测试数据，条件: pk == '{test_pk}'...")
            # 使用 pk 字段进行查询
            results = self.collection.query(
                expr=f"pk == '{test_pk}'",
                output_fields=["pk", "user_id", "role", "timestamp"]
            )
            if results:
                logger.info(f"✅ 查询成功！找到 {len(results)} 条匹配数据。")
                for entity in results:
                    logger.info(f"查询结果: {entity}")
            else:
                logger.warning(f"❌ 未找到 pk 为 '{test_pk}' 的数据。")
            return results
        except MilvusException as me:
            logger.error(f"❌ Milvus 查询测试数据失败: {me}", exc_info=True)
            raise RuntimeError(f"Milvus 查询测试数据失败: {str(me)}")

    def close(self):
        """关闭 Milvus 连接。"""
        try:
            if self.collection:
                self.collection.release() # 释放集合内存
                logger.info(f"✅ 集合 '{self.collection_name}' 已从内存中释放。")
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)
                logger.info("✅ 已断开 Milvus 连接。")
        except MilvusException as me:
            logger.error(f"❌ 关闭 Milvus 连接失败: {me}", exc_info=True)
        except Exception as e:
            logger.error(f"❌ 关闭 Milvus 连接发生意外错误: {e}", exc_info=True)


if __name__ == "__main__":
    milvus_test = MilvusTestManager()
    try:
        # 强制删除并创建集合
        milvus_test.drop_and_create_collection()

        # 准备测试数据
        test_pk = "test_user_001"
        test_embedding = [0.1] * milvus_test.embedding_dim # 使用维度填充测试向量
        test_role = "admin"

        # 插入测试数据
        milvus_test.insert_test_data(test_pk, test_embedding, test_role)

        # 查询测试数据
        milvus_test.query_test_data(test_pk)

    except RuntimeError as e:
        logger.critical(f"脚本执行失败: {e}")
    except Exception as e:
        logger.critical(f"发生意外错误: {e}", exc_info=True)
    finally:
        milvus_test.close()
        logger.info("脚本执行完毕。")
# M:\meeting\core\voice_biometrics\role_manager.py
import numpy as np
from pymilvus import connections, Collection, utility, MilvusException # 导入 MilvusException
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RoleManager:
    """
    管理声纹对应的角色验证。
    """
    def __init__(self):
        # 1. 检查 Milvus 连接是否已存在（由 VoiceprintManager 或其他地方建立）
        if "default" not in connections.list_connections():
            try:
                connections.connect(
                    alias="default", # 统一使用 "default" 别名
                    host=settings.MILVUS_HOST,
                    port=str(settings.MILVUS_PORT) # 端口需要是字符串
                )
                logger.info(f"✅ RoleManager 成功建立到 Milvus 的连接: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            except MilvusException as e:
                logger.error(f"❌ RoleManager 连接 Milvus 失败: {e}", exc_info=True)
                raise RuntimeError(f"RoleManager 无法连接到 Milvus 数据库: {e}")
            except Exception as e:
                logger.error(f"❌ RoleManager 建立 Milvus 连接时发生未知错误: {e}", exc_info=True)
                raise RuntimeError(f"RoleManager 无法连接到 Milvus 数据库: {e}")

        try:
            # 2. 获取集合实例，统一使用 settings.MILVUS_COLLECTION_NAME
            self.collection_name = settings.MILVUS_COLLECTION_NAME
            self.collection = Collection(self.collection_name)
            
            # 3. 移除_check_index_exists和_create_index。
            # 集合和索引的创建及管理责任现在完全由 VoiceprintManager 承担。
            # RoleManager 假设集合及其索引已存在。
            
            # 4. 加载集合到内存
            self.collection.load()
            logger.info(f"🚀 RoleManager 成功加载 Milvus 集合 '{self.collection_name}'。")

        except MilvusException as e:
            logger.error(f"❌ RoleManager 加载 Milvus 集合失败: {e}", exc_info=True)
            raise RuntimeError(f"RoleManager 无法加载 Milvus 集合 '{self.collection_name}': {e}")
        except Exception as e:
            logger.error(f"❌ RoleManager 初始化 Milvus 集合时发生未知错误: {e}", exc_info=True)
            raise RuntimeError(f"RoleManager 无法初始化 Milvus 集合: {e}")

    def verify_role(self, embedding: np.ndarray) -> str:
        """
        验证声纹对应的角色。
        
        Args:
            embedding (np.ndarray): 声纹嵌入向量。
            
        Returns:
            str: 匹配到的角色，如果未找到则返回默认角色。
        """
        if not self.collection:
            logger.error("Milvus 集合未初始化，无法验证角色。")
            return settings.DEFAULT_ROLE

        try:
            # 搜索参数，metric_type 从 settings 中获取
            search_params = {
                "metric_type": settings.MILVUS_INDEX_PARAMS["metric_type"],
                "params": {"nprobe": 16} # nprobe 可以从 settings 中获取或保持默认
            }
            results = self.collection.search(
                data=[embedding.tolist()], # 查询向量必须是列表套列表
                anns_field="embedding",
                param=search_params,
                limit=1, # 只返回最相似的一个
                output_fields=["role"] # 只需返回 role 字段
            )
            
            if results and results[0] and len(results[0]) > 0:
                # 直接通过属性访问 role，更简洁
                role = results[0][0].role 
                logger.info(f"声纹角色验证成功，匹配角色: {role}")
                return role
            else:
                logger.info(f"未找到匹配声纹的角色，返回默认角色: {settings.DEFAULT_ROLE}")
                return settings.DEFAULT_ROLE
        except MilvusException as e:
            logger.error(f"❌ 验证声纹角色时 Milvus 搜索失败: {e}", exc_info=True)
            return settings.DEFAULT_ROLE
        except Exception as e:
            logger.error(f"❌ 验证声纹角色时发生未知错误: {e}", exc_info=True)
            return settings.DEFAULT_ROLE

    def close(self):
        """断开与 Milvus 的连接，并释放资源。"""
        try:
            if self.collection:
                self.collection.release() # 从内存中卸载集合
                logger.info(f"集合 '{self.collection_name}' 已从内存中卸载。")
            # 注意：这里不主动断开连接，因为连接可能被其他组件共享，交由 VoiceprintManager 统一管理。
            # connections.disconnect(alias="default") # 如果确认 RoleManager 是唯一使用连接的地方，可以取消注释
            logger.info("RoleManager 完成操作，连接保持。")
        except Exception as e:
            logger.warning(f"RoleManager 关闭操作时发生错误: {e}", exc_info=True)
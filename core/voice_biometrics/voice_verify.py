# core\voice_biometrics\voice_verify.py

from pymilvus import Collection, connections, MilvusException 

from config.settings import settings
import numpy as np
import logging # 导入日志模块
from typing import List, Dict, Optional 

logger = logging.getLogger(__name__) 

class VoiceVerifier:
    """
    负责从文件路径提取声纹并与Milvus中的声纹进行验证。
    """
    def __init__(self):
        
        
        # 确保 ECAPA 模型已加载并获取维度
        if self.model.classifier is None:
            raise RuntimeError("ECAPA 模型初始化失败，VoiceVerifier 无法加载。")
        self.embedding_dim = self.model.embedding_dim

        # 检查 Milvus 连接是否已存在（由 VoiceprintManager 或其他地方建立）
        # 如果没有名为 "default" 的连接，这里会尝试建立一个
        if "default" not in connections.list_connections():
            try:
                connections.connect(
                    alias="default", # 统一使用 "default" 别名
                    host=settings.MILVUS_HOST,
                    port=str(settings.MILVUS_PORT) # 端口需要是字符串
                )
                logger.info(f"✅ VoiceVerifier 成功建立到 Milvus 的连接: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            except MilvusException as e:
                logger.error(f"❌ VoiceVerifier 连接 Milvus 失败: {e}", exc_info=True)
                raise RuntimeError(f"VoiceVerifier 无法连接到 Milvus 数据库: {e}")
            except Exception as e:
                logger.error(f"❌ VoiceVerifier 建立 Milvus 连接时发生未知错误: {e}", exc_info=True)
                raise RuntimeError(f"VoiceVerifier 无法连接到 Milvus 数据库: {e}")

        try:
            # 获取集合实例，统一使用 settings.MILVUS_COLLECTION_NAME
            self.collection = Collection(settings.MILVUS_COLLECTION_NAME)
            # 加载集合到内存，以便进行搜索
            self.collection.load()
            logger.info(f"🚀 VoiceVerifier 成功加载 Milvus 集合 '{settings.MILVUS_COLLECTION_NAME}'。")
        except MilvusException as e:
            logger.error(f"❌ VoiceVerifier 加载 Milvus 集合失败: {e}", exc_info=True)
            raise RuntimeError(f"VoiceVerifier 无法加载 Milvus 集合 '{settings.MILVUS_COLLECTION_NAME}': {e}")
        except Exception as e:
            logger.error(f"❌ VoiceVerifier 初始化 Milvus 集合时发生未知错误: {e}", exc_info=True)
            raise RuntimeError(f"VoiceVerifier 无法初始化 Milvus 集合: {e}")

    def verify(self, audio_path: str, top_k: int = 3) -> List[Dict]:
        """
        验证声纹并返回匹配结果。
        
        Args:
            audio_path (str): 音频文件路径。
            top_k (int): 返回最相似的 k 个结果。
            
        Returns:
            list: 包含相似用户ID、角色、距离和相似度的字典列表。
                  示例: [{'user_id': 'user123', 'role': 'member', 'distance': 0.1, 'similarity': 0.9}]
        """
        try:
            # 使用 ECAPAWrapper 的 extract_features_from_file 方法
            # 同时传递采样率，确保与模型要求一致
            embedding = self.model.extract_features_from_file(audio_path, settings.VOICE_SAMPLE_RATE)
            if embedding is None:
                logger.warning(f"未能从音频文件 '{audio_path}' 中提取声纹特征。")
                return []
            
            # 从 settings 获取搜索参数，nprobe 可以是一个单独的配置项
            search_params = {
                "metric_type": settings.MILVUS_INDEX_PARAMS["metric_type"], # 从 settings 获取 metric_type
                "params": {"nprobe": 10} # nprobe 可以从 settings 中获取或保持默认
            }
            
            # 执行 Milvus 搜索
            results = self.collection.search(
                data=[embedding.tolist()], # 查询向量必须是列表套列表
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["user_id", "role"] # 现在使用 user_id 和 role
            )
            
            # 解析搜索结果
            verified_results = []
            for hit in results[0]: # results[0] 对应第一个查询向量的结果
                score = 0.0
                if search_params["metric_type"] == "L2":
                    score = 1 / (1 + hit.distance) # L2 距离越小越相似，转换为 0-1 的相似度
                elif search_params["metric_type"] == "COSINE":
                    score = 1 - hit.distance # COSINE 距离越小越相似，转换为 0-1 的相似度

                verified_results.append({
                    "user_id": hit.user_id, # 直接访问 hit.user_id 属性
                    "role": hit.role,     # 直接访问 hit.role 属性
                    "distance": float(hit.distance),
                    "similarity": score
                })
            
            logger.info(f"声纹验证完成，文件 '{audio_path}' 的匹配结果: {verified_results}")
            return verified_results
        except MilvusException as e:
            logger.error(f"❌ 验证声纹时 Milvus 搜索失败: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"❌ 验证声纹时发生未知错误: {e}", exc_info=True)
            return []

    def close(self):
        """断开与 Milvus 的连接，并释放资源。"""
        try:
            if self.collection:
                self.collection.release() # 从内存中卸载集合
                logger.info(f"集合 '{self.collection_name}' 已从内存中卸载。")
            # 注意：这里不主动断开连接，因为连接可能被其他组件共享，交由 VoiceprintManager 统一管理。
            # connections.disconnect(alias="default") # 如果确认 VoiceVerifier 是唯一使用连接的地方，可以取消注释
            logger.info("VoiceVerifier 完成操作，连接保持。")
        except Exception as e:

            logger.warning(f"VoiceVerifier 关闭操作时发生错误: {e}", exc_info=True)

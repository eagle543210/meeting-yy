# services/speaker_recognition_service.py
import logging
import numpy as np
import uuid
import time
from typing import Dict, Any, List, Optional
import os 
from services.voiceprint_service import VoiceprintService 

from core.database.milvus_manager import MilvusManager

from config.settings import settings
from models.role import UserRole 

logger = logging.getLogger(__name__)

class VoiceprintManager:
    """
    声纹管理类，负责协调声纹服务的各种操作，如声纹注册、识别等。
    它接收 VoiceprintService 实例来执行声纹嵌入，并使用 MilvusManager 进行数据库交互。
    """
    def __init__(self, voiceprint_service: VoiceprintService, milvus_manager: MilvusManager): 
        # 1. 注入 VoiceprintService 实例
        self.voiceprint_service = voiceprint_service 

        # 2. 注入 Milvus 数据库管理器实例
        self.milvus_manager = milvus_manager

        logger.info("VoiceprintManager 初始化中...")

        # 检查 VoiceprintService 是否已成功加载其内部模型
        if not self.voiceprint_service.is_model_loaded():
            load_error = self.voiceprint_service.get_load_error()
            logger.critical(f"VoiceprintManager 初始化失败：依赖的 VoiceprintService 模型未加载。错误: {load_error}")
            # 抛出运行时错误，阻止 VoiceprintManager 启动
            raise RuntimeError(f"VoiceprintManager 初始化失败：VoiceprintService 模型未加载。详细: {load_error}")

        # 3. 声纹识别的相似度阈值 (从 settings 获取)
        # 确保 settings.VOICEPRINT_SIMILARITY_THRESHOLD 是一个合适的浮点数
        self.similarity_threshold_l2 = getattr(settings, 'VOICEPRINT_SIMILARITY_THRESHOLD', 0.6) # pyannote通常使用余弦相似度，L2距离阈值可能更小
                                                                                                 # 如果 Milvus 配置是 L2，但 pyannote 内部使用的是余弦相似度并转换为 L2，
                                                                                                 # 那么这个阈值需要根据实际测试来校准。
                                                                                                 # 对于归一化向量，L2 距离与余弦相似度的关系是 `L2_dist^2 = 2 * (1 - cos_sim)`
                                                                                                 # 如果 pyannote 默认相似度阈值是 0.7（余弦），那么 L2 距离大约是 sqrt(2 * (1 - 0.7)) = sqrt(0.6) ~= 0.77
                                                                                                 # 这里先给一个经验值，但你可能需要根据实际情况调整。
        logger.info(f"声纹相似度阈值 (L2 距离): {self.similarity_threshold_l2}")

        # 4. 缓存已注册发言人的ID和角色 (减少 Milvus 交互)
        self.registered_speakers: Dict[str, Dict[str, Any]] = {}
        # 注意：这里假设 milvus_manager.load_all_voice_prints 是同步方法。
        # 如果它是异步的，你不能直接在这里 await。
        # 一个更好的模式是让 VoiceprintManager 有一个异步的 `async init_data()` 方法，
        # 并在 BackendCoordinator 中 await 调用它。
        # 但为了避免大的结构性改动，暂时保持现状，并在后续如果报错再调整。
        try:
            # 假定 load_all_voice_prints 是同步方法。如果不是，这里会出错。
            loaded_speakers_dict = self.milvus_manager.load_all_voice_prints()
            
            for user_id, speaker_data in loaded_speakers_dict.items():
                role = speaker_data.get('role')
                if user_id: 
                    self.registered_speakers[user_id] = {'user_id': user_id, 'role': role}
            
            logger.info(f"VoiceprintManager 已加载 {len(self.registered_speakers)} 个已注册声纹的元数据。")
        except Exception as e:
            logger.error(f"加载所有声纹元数据失败: {e}", exc_info=True)
            logger.warning("声纹识别功能可能受影响，因为无法加载所有注册声纹。")

        # 5. 添加一个简短的缓存，用于防止短时间内同一ID重复出现新用户提示
        self.recent_speaker_ids_cache: Dict[str, float] = {} # key: raw_user_id, value: timestamp of last appearance
        self.cache_duration_sec: float = 2.0 # 缓存2秒，防止频繁重复判断

        logger.info("VoiceprintManager 初始化完成。")

    async def identify_speaker(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        根据音频数据识别说话人。
        如果能匹配到现有声纹，则返回其ID；否则注册为新用户（或标记为未知）。
        Args:
            audio_data (np.ndarray): 音频的 numpy 数组数据。
            sample_rate (int): 音频的采样率。
        Returns:
            Dict[str, Any]: 包含识别结果的字典。
        """
        current_time = time.time()

        try:
            # 1. 提取当前音频的声纹特征，通过 VoiceprintService 异步完成
            # VoiceprintService 的 get_speaker_embedding 现在应该直接接受 numpy 数组和采样率
            current_embedding_np = await self.voiceprint_service.get_speaker_embedding(audio_data, sample_rate)
            
            # --- 关键检查：确保嵌入中不含 NaN 或 Inf 值 ---
            if current_embedding_np is None:
                logger.warning("未能从音频数据中提取声纹特征，返回未知发言人。")
                return {"user_id": "未知发言人", "role": UserRole.UNKNOWN.value, "confidence": 0.0, "status": "failed_extraction"}
                
            # 检查是否有 NaN/Inf，如果存在，立即返回错误
            if np.isnan(current_embedding_np).any() or np.isinf(current_embedding_np).any():
                logger.error("❌ 提取的声纹特征包含 NaN 或 Inf 值，跳过处理并返回未知发言人。")
                return {"user_id": "未知发言人", "role": UserRole.UNKNOWN.value, "confidence": 0.0, "status": "invalid_embedding"}
                
            # 将 numpy array 转换为 Python list，这是 Milvus 插入和查询向量所需的格式
            current_embedding_list = current_embedding_np.tolist() 

            logger.debug(f"提取到声纹特征，形状: {current_embedding_np.shape}, 前5个值: {current_embedding_list[:5]}")

            # 2. 在 Milvus 中搜索最相似的声纹
            search_results = await self.milvus_manager.search_voice_prints(current_embedding_list, top_k=1)
            
            best_match_id = None
            best_match_distance = float('inf')
            best_match_role = UserRole.UNKNOWN.value 

            if search_results:
                best_match = search_results[0]
                best_match_id = best_match.get("user_id")
                best_match_distance = best_match.get("distance")
                best_match_role = best_match.get("role", UserRole.REGISTERED_USER.value) 

                logger.debug(f"Milvus 找到最相似的ID: {best_match_id}, 距离: {best_match_distance:.4f}, 角色: {best_match_role}")

            # 3. 根据距离阈值判断是否匹配
            if best_match_id and best_match_distance <= self.similarity_threshold_l2:
                user_id = best_match_id
                role = best_match_role
                
                self.recent_speaker_ids_cache[user_id] = current_time

                # 置信度计算：距离越小，置信度越高。对于 L2 距离。
                # 假设 threshold 是 L2 距离的最大可接受值
                if self.similarity_threshold_l2 > 0:
                    # 将距离归一化到 [0, 1] 范围，0表示完全匹配，1表示刚好在阈值
                    normalized_distance = best_match_distance / self.similarity_threshold_l2
                    # 置信度：1 - normalized_distance (距离越小，置信度越高)
                    confidence = max(0.0, 1 - normalized_distance)
                else:
                    confidence = 0.0 
                confidence = max(0.0, min(1.0, confidence)) # 确保在 0 到 1 之间

                logger.info(f"✅ 匹配到现有用户: {user_id}, 距离: {best_match_distance:.4f}, 置信度: {confidence:.4f}, 角色: {role}")
                return {"user_id": user_id, "role": role, "confidence": confidence, "status": "matched"}
            else:
                # 如果没有匹配到，或者相似度低于阈值，则标记为新用户
                new_user_id = str(uuid.uuid4())
                new_role = UserRole.GUEST.value 

                # 清理过期缓存
                for cached_id, last_seen_time in list(self.recent_speaker_ids_cache.items()):
                    if current_time - last_seen_time >= self.cache_duration_sec:
                        del self.recent_speaker_ids_cache[cached_id] 
                
                # 存储新的声纹到 Milvus
                await self.milvus_manager.insert_voice_print(new_user_id, current_embedding_list, new_role)
                # 更新本地缓存
                self.registered_speakers[new_user_id] = {"user_id": new_user_id, "role": new_role} 
                
                self.recent_speaker_ids_cache[new_user_id] = current_time

                # 如果没有匹配到，通常置信度为0
                logger.info(f"🆕 注册新发言人: {new_user_id}, 距离: {best_match_distance:.4f} (未匹配阈值), 角色: {new_role}")
                return {"user_id": new_user_id, "role": new_role, "confidence": 0.0, "status": "new_user"}

        except Exception as e:
            logger.error(f"❌ 声纹识别或管理失败: {e}", exc_info=True)
            return {"user_id": "服务错误", "role": UserRole.ERROR.value, "confidence": 0.0, "status": "service_error"}

    async def register_speaker_with_id_and_audio_file(self, user_id: str, audio_file_path: str) -> bool:
        """
        根据提供的用户ID和音频文件路径注册声纹。
        主要用于预注册已知用户。
        """
        logger.info(f"正在为用户 {user_id} 注册声纹 (通过文件)...")
        # 直接使用 VoiceprintService 的注册方法，它应该处理了嵌入提取和 Milvus 存储
        # VoiceprintService.register_voiceprint 应该接受 user_id 和 file_path
        success = await self.voiceprint_service.register_voiceprint(user_id, audio_file_path)
        if success:
            # 如果 VoiceprintService 注册成功，更新本地缓存
            self.registered_speakers[user_id] = {"user_id": user_id, "role": UserRole.REGISTERED_USER.value}
            logger.info(f"用户 {user_id} 声纹注册成功。")
        else:
            logger.error(f"用户 {user_id} 声纹注册失败。")
        return success

    def close(self):
        """关闭 VoiceprintManager 依赖的服务连接。"""
        logger.info("正在关闭 VoiceprintManager 依赖的服务...")
        try:
            # MilvusManager 的 close 方法通常是同步的，所以这里不需要 await
            self.milvus_manager.close()
            logger.info("MilvusManager 连接已关闭。")
        except Exception as e:
            logger.error(f"关闭 MilvusManager 连接时发生错误: {e}", exc_info=True)

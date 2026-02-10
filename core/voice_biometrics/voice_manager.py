# M:\meeting\core\voice_biometrics\voice_manager.py

import logging
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncio

from config.settings import settings
from services.mongodb_manager import MongoDBManager # 确保导入 MongoDBManager
from core.database.milvus_manager import MilvusManager
from services.voiceprint_service import VoiceprintService # 导入 VoiceprintService
from models import User, UserRole

logger = logging.getLogger(__name__)

class VoiceprintManager:
    """
    负责管理声纹注册、识别和用户信息的类。
    它协调 VoiceprintService (用于声纹提取) 和 MilvusManager (用于向量搜索)。
    """
    def __init__(self, 
                 voiceprint_service: VoiceprintService,
                 milvus_manager: MilvusManager,
                 mongo_service: MongoDBManager, # <-- 关键修正：在这里添加 mongo_service 参数
                 settings_obj: settings # 接收 settings_obj
                ):
        logger.info("VoiceprintManager 初始化中...")
        self.voiceprint_service = voiceprint_service
        self.milvus_manager = milvus_manager
        self.mongo_service = mongo_service # 存储 MongoDBManager 实例
        self.settings = settings_obj # 存储 settings_obj

        # 存储已注册说话人的本地缓存 {user_id: {username: str, role: UserRole, embedding: List[float]}}
        self.registered_speakers: Dict[str, Dict[str, Any]] = {}
        self.is_initialized_successfully: bool = False # 标记是否成功加载了初始数据

        logger.info("VoiceprintManager 属性已设置，等待加载初始数据。")

    async def load_initial_data(self):
        """
        从 MongoDB 加载所有已注册的用户数据到内存缓存。
        """
        logger.info("VoiceprintManager: 正在加载初始用户数据...")
        try:
            users = await self.mongo_service.get_all_users()
            self.registered_speakers.clear() # 清空现有缓存
            for user in users:
                # 假设用户对象中包含 voice_embedding 字段
                if user.voice_embedding:
                    self.registered_speakers[user.user_id] = {
                        "username": user.username,
                        "role": user.role.value, # 存储为字符串
                        "embedding": user.voice_embedding # 存储嵌入
                    }
            self.is_initialized_successfully = True
            logger.info(f"VoiceprintManager: 已成功加载 {len(self.registered_speakers)} 个注册用户。")
        except Exception as e:
            self.is_initialized_successfully = False
            logger.critical(f"VoiceprintManager: 加载初始用户数据失败: {e}", exc_info=True)
            raise RuntimeError(f"VoiceprintManager 加载初始数据失败: {e}") from e

    async def register_voice(self, audio_data_np: np.ndarray, sample_rate: int, user_name: str, role_str: str) -> Dict[str, Any]:
        """
        注册用户的声纹和信息。
        Args:
            audio_data_np (np.ndarray): 音频数据。
            sample_rate (int): 音频采样率。
            user_name (str): 用户名。
            role_str (str): 用户角色字符串。
        Returns:
            Dict[str, Any]: 注册结果。
        Raises:
            ValueError: 如果无法生成声纹嵌入或声纹冲突。
        """
        logger.info(f"VoiceprintManager: 尝试注册用户 '{user_name}' (角色: {role_str}) 的声纹...")
        
        try:
            role = UserRole(role_str)
        except ValueError:
            raise ValueError(f"无效的用户角色: {role_str}")

        # 1. 从 VoiceprintService 获取声纹嵌入
        voice_embedding_list = await self.voiceprint_service.get_speaker_embedding(audio_data_np, sample_rate)
        if voice_embedding_list is None: # <-- 修正：将 === None 改为 is None
            raise ValueError("无法从提供的音频生成声纹嵌入。请确保音频质量和时长符合要求。")
        
        # 检查嵌入向量是否有效
        voice_embedding_np = np.array(voice_embedding_list)
        if np.isnan(voice_embedding_np).any() or np.isinf(voice_embedding_np).any():
            logger.error("❌ 提取的注册声纹特征包含 NaN 或 Inf 值，阻止注册。")
            raise ValueError("提取的声纹特征无效。")

        # 2. 搜索 Milvus 检查是否存在相似声纹
        search_results = await self.milvus_manager.search_voice_prints(voice_embedding_list, top_k=1)

        existing_milvus_user_id: Optional[str] = None
        best_match_distance = float('inf')
        
        if search_results and len(search_results) > 0:
            best_match = search_results[0]
            existing_milvus_user_id = best_match.get("user_id")
            best_match_distance = best_match.get("distance")

        # 3. 判断是否为现有用户，或者是否冲突
        if existing_milvus_user_id and best_match_distance < self.settings.VOICEPRINT_SIMILARITY_THRESHOLD:
            # 找到相似声纹，尝试获取 MongoDB 中的用户信息
            existing_user_in_mongo = await self.mongo_service.find_user_by_id(existing_milvus_user_id)
            
            if existing_user_in_mongo:
                # 如果找到同 ID 的用户，且用户名也匹配，则更新其角色和嵌入（如果需要）
                if existing_user_in_mongo.username == user_name:
                    logger.info(f"用户 '{user_name}' (ID: {existing_milvus_user_id}) 声纹已存在，正在更新角色为 '{role.value}'。")
                    existing_user_in_mongo.role = role
                    existing_user_in_mongo.last_active = datetime.utcnow()
                    existing_user_in_mongo.voice_embedding = voice_embedding_list # 更新嵌入
                    await self.mongo_service.save_user(existing_user_in_mongo)
                    
                 
                    
                    # 更新本地缓存
                    self.registered_speakers[existing_milvus_user_id] = {
                        "username": user_name,
                        "role": role.value,
                        "embedding": voice_embedding_list
                    }
                    return {"status": "registered", "message": "声纹已存在，用户信息已更新。", "user_id": existing_milvus_user_id, "user_name": user_name, "role": role.value}
                else:
                    # 声纹冲突：找到相似声纹但用户名不同
                    logger.warning(f"声纹与现有用户 '{existing_user_in_mongo.username}' (ID: {existing_milvus_user_id}) 冲突，但尝试注册名为 '{user_name}'。")
                    raise ValueError(f"声纹与现有用户 '{existing_user_in_mongo.username}' 冲突，请检查或使用不同音频。")
            else:
                # Milvus 中有声纹但 MongoDB 中没有对应用户，这可能是数据不一致。
                # 为了注册流程继续，我们将创建一个新用户ID并插入新声纹。
                logger.warning(f"Milvus 中存在 ID '{existing_milvus_user_id}' 的声纹，但 MongoDB 中无对应用户。将注册为新用户并插入新声纹。")
                # 理论上，这里可以删除旧的 Milvus 向量 `await self.milvus_manager.delete_voice_print(existing_milvus_user_id)`
                # 但为了简化，暂时直接注册新的，旧向量可能变为“孤儿”。
        
        # 4. 注册新用户或处理 Milvus 有而 Mongo 无的情况
        new_user_id = str(uuid.uuid4())
        
        new_user = User(
            user_id=new_user_id,
            username=user_name,
            role=role,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            voice_embedding=voice_embedding_list # 保存声纹嵌入到用户模型
        )
        
        await self.mongo_service.save_user(new_user)
        logger.info(f"用户 '{user_name}' (ID: {new_user_id}) 已保存到 MongoDB。")

        # 将声纹嵌入插入 Milvus
        insert_milvus_result = await self.milvus_manager.insert_voice_print(new_user_id, voice_embedding_list, role.value)
        logger.info(f"声纹 ID '{new_user_id}' 已插入到 Milvus。插入结果: {insert_milvus_result}")

        # 更新本地缓存
        self.registered_speakers[new_user_id] = {
            "username": user_name,
            "role": role.value,
            "embedding": voice_embedding_list
        }

        return {"status": "registered", "message": "声纹注册成功。", "user_id": new_user_id, "user_name": user_name, "role": role.value}

    async def identify_speaker(self, audio_data_np: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """
        识别音频片段中的说话人。
        Args:
            audio_data_np (np.ndarray): 音频数据。
            sample_rate (int): 音频采样率。
        Returns:
            Optional[Dict[str, Any]]: 识别到的说话人信息 (user_id, username, role, distance)，如果未识别到则为 None。
        """
        if not self.voiceprint_service or not self.milvus_manager or not self.mongo_service:
            logger.error("识别说话人所需服务未完全初始化。")
            return None

        # 1. 从 VoiceprintService 获取声纹嵌入
        voice_embedding_list = await self.voiceprint_service.get_speaker_embedding(audio_data_np, sample_rate)
        if voice_embedding_list is None: # <-- 修正：将 === None 改为 is None
            logger.warning("无法从提供的音频生成声纹嵌入，跳过说话人识别。")
            return None

        # 检查嵌入向量是否有效
        voice_embedding_np = np.array(voice_embedding_list)
        if np.isnan(voice_embedding_np).any() or np.isinf(voice_embedding_np).any():
            logger.warning("提取的声纹特征包含 NaN 或 Inf 值，跳过识别。")
            return None

        # 2. 在 Milvus 中搜索最相似的声纹
        search_results = await self.milvus_manager.search_voice_prints(voice_embedding_list, top_k=1)

        if search_results and len(search_results) > 0:
            best_match = search_results[0]
            matched_user_id = best_match.get("user_id")
            distance = best_match.get("distance")

            if distance < self.settings.VOICEPRINT_SIMILARITY_THRESHOLD:
                # 3. 从 MongoDB 获取匹配的用户信息
                user_info = await self.mongo_service.find_user_by_id(matched_user_id)
                if user_info:
                    logger.info(f"识别到说话人: ID={matched_user_id}, 用户名={user_info.username}, 角色={user_info.role.value}, 距离={distance:.4f}")
                    return {
                        "user_id": matched_user_id,
                        "username": user_info.username,
                        "role": user_info.role.value,
                        "distance": distance
                    }
                else:
                    logger.warning(f"Milvus 匹配到 ID '{matched_user_id}'，但在 MongoDB 中未找到对应用户。")
                    return None
            else:
                logger.info(f"未识别到匹配说话人 (最佳匹配距离 {distance:.4f} 超出阈值 {self.settings.VOICEPRINT_SIMILARITY_THRESHOLD})。")
                return None
        else:
            logger.info("Milvus 中未找到任何匹配的声纹。")
            return None

    async def close(self):
        """
        关闭 VoiceprintManager，释放资源。
        """
        logger.info("Closing VoiceprintManager...")
        # VoiceprintManager 不直接管理 VoiceprintService 或 MilvusManager 的生命周期，
        # 它们由 BackendCoordinator 或 lifespan 管理。
        self.registered_speakers.clear()
        self.is_initialized_successfully = False
        logger.info("VoiceprintManager closed.")

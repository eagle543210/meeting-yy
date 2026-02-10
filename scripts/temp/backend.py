# backend.py
import logging
import numpy as np # 用于处理音频数据
from typing import Dict, Any, Optional
import uuid # 用于生成唯一 ID
import asyncio # 新增，因为我们将在处理语音流时使用异步方法

# 导入核心模块
from core.knowledge_engine import KnowledgeProcessor
from core.decision_system import DecisionEngine
from core.data_processing import (
    DataFlowHandler,
    TopicTracker
)
# 导入权限检查模块和相关模型
from auth.check_permission import check_permission, get_current_user # 导入 get_current_user
from models.user import User # 导入 User 模型 (用于类型提示和操作)
from models.role import UserRole # 导入 UserRole 枚举
from models.permission import Permission # 导入 Permission 枚举

# 导入服务
from services import MongoLogger
# 导入声纹管理核心类
from core.voice_biometrics.voice_manager import VoiceprintManager # 统一使用 VoiceprintManager
from core.speech_to_text.stt_processor import SpeechToTextProcessor
from core.knowledge_engine.kg_builder import KnowledgeGraphBuilder # 如果需要 KGBuilder 实例
from config.settings import settings # 导入 settings 以获取 VOICE_SAMPLE_RATE

logger = logging.getLogger(__name__)

class BackendCoordinator:
    def __init__(self):
        # 统一使用 VoiceprintManager 处理所有声纹相关操作
        self.voice_manager = VoiceprintManager()
        
        # 核心功能模块的初始化，避免重复创建实例
        self.knowledge_processor = KnowledgeProcessor()
        self.decision_engine = DecisionEngine()
        self.data_flow_handler = DataFlowHandler()
        self.topic_tracker = TopicTracker()
        self.mongo_logger = MongoLogger() # 更具描述性的名称
        
        # 如果需要知识图谱构建器，在此处初始化
        # self.kg_builder = KnowledgeGraphBuilder() 
        self.speaker_id_to_name_map: Dict[str, str] = {} 
        self.next_unregistered_speaker_index = 1 
        self.recent_unknown_ids_cache: Dict[str, str] = {}
        self.cache_size = 100 
        # --- 新增：发言人 ID 映射和计数器 ---
        # 用于存储原始 speaker_id (UUID) 到友好名称的映射
        self.speaker_id_to_name_map: Dict[str, str] = {} 
        # 用于为未注册的 ID 生成递增的友好名称
        self.next_unregistered_speaker_index = 1 
        # 用于缓存最近识别的未知 ID 及其分配的友好名称，以减少重复分配
        self.recent_unknown_ids_cache: Dict[str, str] = {}
        self.cache_size = 100 # 缓存最近 100 个未知 ID
        # --- 结束新增 ---
        self.stt_processor = None # 初始化为 None
        if settings.ENABLE_STT: # 根据 settings 中的配置决定是否启用 STT
            logger.info("初始化语音转文本模型...")
            try:
                self.stt_processor = SpeechToTextProcessor() # 实例化 STT 处理器
                logger.info("语音转文本模型初始化成功。")
            except Exception as e:
                logger.error(f"语音转文本模型初始化失败: {e}", exc_info=True)
                self.stt_processor = None # 如果初始化失败，禁用 STT
        else:
            logger.info("settings.ENABLE_STT 为 False，语音转文本功能已禁用。")
        logger.info("BackendCoordinator 已初始化，所有核心服务已加载。")

    async def register_voice(self, audio_data: np.ndarray, user_name: str, role: str) -> Dict[str, Any]:
        """
        新增声纹注册接口。
        
        Args:
            audio_data (np.ndarray): 音频数据 (例如，float32 的 NumPy 数组)。
            user_name (str): 用户提供的友好名称。
            role (str): 注册时指定的用户角色。
        """
        if not isinstance(audio_data, np.ndarray) or audio_data.dtype != np.float32:
            logger.error("音频数据格式不正确，应为 float32 的 NumPy 数组。")
            return {"status": "error", "message": "音频数据格式不正确。"}

        # 1. 提取声纹特征
        # 假设 VoiceprintManager 的 ecapa_model 有 extract_features_from_buffer 方法
        embedding = self.voice_manager.ecapa_model.extract_features_from_buffer(audio_data, settings.VOICE_SAMPLE_RATE)
        if embedding is None:
            logger.error("声纹注册失败: 无法从音频数据中提取特征。")
            return {"status": "error", "message": "无法提取声纹特征。"}
        
        # 2. 生成新用户 ID (通常是 UUID)
        user_id = str(uuid.uuid4()) 
        
        # 3. 将声纹和用户信息插入 Milvus 集合
        try:
            insert_data = [{
                "user_id": user_id,
                "embedding": embedding.tolist(), # Milvus 接受列表形式
                "role": role # 使用传入的角色
            }]
            self.voice_manager.milvus_collection.insert(insert_data)
            self.voice_manager.milvus_collection.flush() # 刷新确保数据写入

            # 4. 在你的用户数据库中创建用户记录
            new_user = User(user_id=user_id, username=user_name, role=UserRole(role)) # 使用传入的 user_name
            await new_user.save() # 模拟保存到用户数据库

            # --- 新增：将注册的原始 ID 映射到用户提供的名称 ---
            self.speaker_id_to_name_map[user_id] = user_name
            logger.info(f"声纹注册：原始ID '{user_id}' 映射到名称 '{user_name}'。")
            # --- 结束新增 ---

            logger.info(f"✅ 成功注册新声纹用户。User ID: {user_id}, 用户名: {user_name}, 角色: {role}。")
            return {
                "status": "registered",
                "user_id": user_id, # 返回原始的 Milvus ID，前端需要
                "user_name": user_name, # 返回注册的名称，方便前端显示
                "role": role,
                "message": "声纹注册成功"
            }
        except Exception as e:
            logger.error(f"❌ 声纹注册失败: {e}", exc_info=True)
            return {"status": "error", "message": f"声纹注册失败: {e}"}

    async def process_audio_for_websocket(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        处理WebSocket接收到的音频，进行声纹识别和名称映射。
        这个方法将替代直接在 app.py 中调用 voice_manager.process_audio 的部分。
        """
        speaker_result = await self.voice_manager.process_audio(audio_data, sample_rate)

        identified_speaker_id = speaker_result.get("user_id", "unknown")
        identified_role = speaker_result.get("role", UserRole.GUEST.value) # 确保是字符串

        # --- 新增逻辑：处理 speaker_id 映射到友好名称 ---
        friendly_display_name = self.speaker_id_to_name_map.get(identified_speaker_id)

        if friendly_display_name:
            # 如果是注册用户且有映射，直接使用映射的名称
            pass
        elif identified_speaker_id != "unknown":
            # 对于 Milvus 返回的未知 ID (通常是 UUID 形式)
            # 检查是否在最近的未知 ID 缓存中，以保持短时间内的新 ID一致性
            if identified_speaker_id in self.recent_unknown_ids_cache:
                friendly_display_name = self.recent_unknown_ids_cache[identified_speaker_id]
            else:
                # 分配一个新的“新用户-X”名称
                friendly_display_name = f"新用户-{self.next_unregistered_speaker_index}"
                self.next_unregistered_speaker_index += 1
                # 更新映射和缓存
                self.speaker_id_to_name_map[identified_speaker_id] = friendly_display_name
                self.recent_unknown_ids_cache[identified_speaker_id] = friendly_display_name
                # 保持缓存大小
                if len(self.recent_unknown_ids_cache) > self.cache_size:
                    # 移除最老的缓存项 (这里简化为随机移除，实际可以实现 LRU)
                    oldest_key = next(iter(self.recent_unknown_ids_cache))
                    del self.recent_unknown_ids_cache[oldest_key]
                logger.info(f"分配新用户名称：原始ID '{identified_speaker_id}' 映射到名称 '{friendly_display_name}'。")
        else:
            # 如果声纹识别直接返回 "unknown"，则显示 "未知发言人"
            friendly_display_name = "未知发言人"

        # 返回处理后的结果，其中 user_id 是友好的名称
        return {
            "user_id": friendly_display_name, # 这是前端最终会看到的友好名称
            "original_speaker_id": identified_speaker_id, # 原始 Milvus ID，如果需要内部记录
            "role": identified_role,
            "confidence": speaker_result.get("confidence", 0.0),
            "status": speaker_result.get("status") # 可能是 "identified", "registered", "unregistered"
        }
        # --- 结束新增逻辑 ---

    async def process_meeting(self, audio_data: np.ndarray, transcript: str) -> Dict[str, Any]:
        """
        处理会议数据，参数列表与调用一致。
        
        Args:
            audio_data (np.ndarray): 会议音频数据 (例如，float32 的 NumPy 数组)。
            transcript (str): 会议文字稿。
        """
        if not isinstance(audio_data, np.ndarray) or audio_data.dtype != np.float32:
            logger.error("会议音频数据格式不正确，应为 float32 的 NumPy 数组。")
            raise ValueError("会议音频数据格式不正确。")

        meeting_data = {
            'audio': audio_data, # 现在是 NumPy 数组
            'transcript': transcript,
            'speeches': [] # TODO: 这应该通过语音分离（Diarization）步骤填充
        }
        
        # --- 语音分离和说话人识别 ---
        # 这是一个关键步骤，但超出了当前代码的范围。
        # 实际流程会是：整个 audio_data -> 语音分离模块 -> 返回多个 (segment_audio, speaker_id, start_time, end_time)
        # 然后对每个 segment_audio 调用 voice_manager.process_audio。
        
        # 为演示目的，我们模拟一个简单的说话人识别结果
        logger.info("正在对会议音频进行模拟说话人识别...")
        
        # !!! 注意：这里也需要调用我们新的 process_audio_for_websocket 方法来获取友好名称
        # 虽然这里是离线处理，但为了统一逻辑和获取友好名称，可以复用。
        # 如果 process_audio_for_websocket 做了过多WebSocket特有逻辑，
        # 则需要在 BackendCoordinator 中实现一个通用的 get_speaker_info 方法，
        # 让 register_voice 和 process_audio_for_websocket 都调用它。
        # 但目前为了简化，我们在这里直接使用 process_audio_for_websocket 来获取映射后的 ID
        speaker_info_for_meeting = await self.process_audio_for_websocket(
            meeting_data['audio'], settings.VOICE_SAMPLE_RATE
        )
        
        speakers_identified: Dict[str, UserRole] = {} # 存储 {user_id: UserRole}
        if speaker_info_for_meeting and speaker_info_for_meeting.get("status") in ["identified", "registered"]:
            # 使用 friendly_display_name 作为 key
            friendly_user_id = speaker_info_for_meeting.get("user_id") 
            role_str = speaker_info_for_meeting.get("role")
            if friendly_user_id and role_str:
                speakers_identified[friendly_user_id] = UserRole(role_str) # 转换为 UserRole 枚举
                logger.info(f"模拟识别主要说话人: ID={friendly_user_id}, 角色={role_str}")
        else:
            logger.warning("未能识别主要说话人，将使用默认访客角色。")
            speakers_identified["未知发言人"] = UserRole.GUEST # 使用友好名称

        # --- 议题权重计算 ---
        # 假设 meeting_data['speeches'] 会被填充，其中包含 speaker_id
        # 例如: [{'topic': '议题A', 'speaker_id': 'user_id_from_diarization', 'duration': 60, 'votes': 5}]
        for speech in meeting_data.get('speeches', []): 
            speaker_id_in_speech = speech.get('speaker_id')
            # 从识别的说话人列表中获取角色，如果没有识别到则默认为 GUEST
            # TODO: 这里需要更复杂的逻辑来将 speech_id 映射到 friendly_user_id，
            # 因为 diarization 可能会返回不同的临时 ID。
            # 目前暂时使用 speakers_identified 中的 key (友好名称)
            speaker_role = speakers_identified.get(speaker_id_in_speech, UserRole.GUEST) 
            
            self.topic_tracker.update_weights({
                'topic': speech['topic'],
                'speaker_role': speaker_role, 
                'duration': speech['duration'],
                'vote_count': speech.get('votes', 0)
            })
        
        # --- 知识处理 ---
        triples = self.knowledge_processor.extract_triples(meeting_data['transcript'])
        
        # --- 数据流处理 ---
        summary, actions = self.data_flow_handler.generate_minutes(
            meeting_data['transcript'],
            self.topic_tracker.get_priority_topics()
        )
        
        # --- 决策支持 ---
        decision = self.decision_engine.evaluate({
            'text': meeting_data['transcript'],
            'topics': self.topic_tracker.topic_weights,
            'speakers': speakers_identified # 传入已识别的说话人 (友好名称)
        })
        
        # --- 日志记录 ---
        meeting_id = str(uuid.uuid4()) 
        self.mongo_logger.log_decision(
            meeting_id,
            {
                'decision': decision,
                'actions': actions,
                'topics': self.topic_tracker.get_priority_topics(),
                'identified_speakers': {uid: role.value for uid, role in speakers_identified.items()} # 记录为字符串 (友好名称)
            }
        )
        
        logger.info(f"会议 {meeting_id} 处理完成。")
        return {
            'summary': summary,
            'actions': actions,
            'decision': decision,
            'identified_speakers': {uid: role.value for uid, role in speakers_identified.items()} # 返回为字符串 (友好名称)
        }

    def query_knowledge_graph(self, entity: str, depth: int = 1) -> Dict[str, Any]:
        """
        查询知识图谱（同步方法）
        
        Args:
            entity (str): 要查询的实体名称。
            depth (int): 查询深度 (默认1)。
        """
        try:
            return self.knowledge_processor.query(entity, depth)
        except Exception as e:
            logger.error(f"知识图谱查询失败 (实体: '{entity}', 深度: {depth}): {str(e)}", exc_info=True)
            raise RuntimeError(f"知识图谱查询失败: {e}")

    @check_permission(Permission.EDIT_ROLES)
    async def update_user_role(self, voiceprint_id: str, new_role: UserRole) -> Dict[str, Any]:
        """
        更新用户角色。此方法需要当前用户拥有 EDIT_ROLES 权限。
        
        Args:
            voiceprint_id (str): 用户的声纹ID (通常对应 Milvus 中的 user_id)。
            new_role (UserRole): 要设置的新角色。
        """
        user = await User.find_by_voiceprint(voiceprint_id) 
        if not user:
            logger.warning(f"更新角色失败: 未找到声纹 ID 为 '{voiceprint_id}' 的用户。")
            raise ValueError(f"用户不存在: 无法找到声纹 ID 为 '{voiceprint_id}' 的用户。")
        
        old_role = user.role
        if old_role == new_role:
            logger.info(f"用户 '{user.username}' (ID: {user.user_id}) 的角色已经是 '{new_role.value}'，无需更新。")
            return {"status": "no_change", "user_id": user.user_id, "new_role": new_role.value, "message": "角色未改变"}

        user.role = new_role
        await user.save() # 模拟保存到用户数据库

        # --- 新增：更新 speaker_id_to_name_map 中映射的友好名称 ---
        # 假设 user.username 已经是友好名称，并且 voiceprint_id 就是原始 Milvus ID
        if voiceprint_id in self.speaker_id_to_name_map:
            self.speaker_id_to_name_map[voiceprint_id] = user.username # 确保映射是最新的
        # --- 结束新增 ---

        try:
            logger.info(f"✅ 成功更新用户 '{user.username}' (ID: {user.user_id}) 的角色从 '{old_role.value}' 到 '{new_role.value}'。")
            return {"status": "success", "user_id": user.user_id, "new_role": new_role.value, "message": "角色更新成功"}
        except Exception as e:
            logger.error(f"❌ 更新用户 '{user.username}' (ID: {user.user_id}) Milvus 角色失败: {e}", exc_info=True)
            raise RuntimeError(f"更新用户 Milvus 角色失败: {e}")
        
    @check_permission(Permission.EXPORT_MINUTES)
    async def export_meeting_minutes(self, meeting_id: str) -> Dict[str, Any]:
        """导出会议纪要。此方法需要当前用户拥有 EXPORT_MINUTES 权限。"""
        minutes = self.data_flow_handler.generate_minutes(meeting_id)
        logger.info(f"会议 {meeting_id} 纪要导出请求完成。")
        return {
            "content": minutes,
            "format": "docx", 
            "status": "success"
        }
    
    @check_permission(Permission.EXPORT_USER_SPEECH)
    async def export_user_speeches(self, meeting_id: str, user_id: str) -> Dict[str, Any]:
        """导出特定用户发言。此方法需要当前用户拥有 EXPORT_USER_SPEECH 权限。"""
        # TODO: 这里 user_id 可能是友好名称，你需要将其映射回原始 Milvus ID 来查询 MongoLogger
        # 如果 MongoLogger 存储的是原始 ID，你需要：
        # 1. 维护一个反向映射 (友好名称 -> 原始 ID)
        # 2. 或者在 MongoLogger 内部提供按友好名称查询的能力
        # 假设 speeches 结构已经包含友好名称，或者 MongoLogger 能够处理映射
        speeches = self.mongo_logger.get_user_speeches(meeting_id, user_id) 
        logger.info(f"会议 {meeting_id} 中用户 {user_id} 的发言导出请求完成。")
        return {
            "user_id": user_id,
            "speeches": speeches,
            "count": len(speeches),
            "status": "success"
        }
    
    @check_permission(Permission.GENERATE_REPORTS)
    async def generate_summary_report(self, meeting_id: str) -> Dict[str, Any]:
        """生成总结报告。此方法需要当前用户拥有 GENERATE_REPORTS 权限。"""
        report = self.decision_engine.generate_report(meeting_id)
        logger.info(f"会议 {meeting_id} 总结报告生成请求完成。")
        return {
            "report": report,
            "status": "success"
        }
    
    def calculate_speech_weight(self, speaker_role: UserRole) -> float:
        """根据角色计算发言权重。"""
        return settings.ROLE_WEIGHTS.get(speaker_role.value, 1.0) # 使用 .value 获取枚举的字符串值

    def close(self):
        """关闭所有依赖的服务和资源。"""
        logger.info("正在关闭 BackendCoordinator 依赖的服务...")
        try:
            self.voice_manager.close() # 关闭 Milvus 连接等
            # 如果其他组件有 close 方法，也在此调用
            # self.knowledge_processor.close()
            # self.decision_engine.close()
            # self.data_flow_handler.close()
            # self.mongo_logger.close()
            logger.info("BackendCoordinator 依赖的服务已关闭。")
        except Exception as e:
            logger.error(f"关闭 BackendCoordinator 服务时发生错误: {e}", exc_info=True)
# core\data_processing\minute_generator.py

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np
import uuid # 用于生成唯一ID
import os # 用于文件操作
import soundfile as sf # 用于保存音频文件

# 导入服务和模型
from config.settings import settings
from services.mongodb_manager import MongoDBManager 
from services.llm_service import LLMModel
from services.embedding_service import BGEEmbeddingModel
from services.neo4j_service import Neo4jService 
from services.voiceprint_service import VoiceprintService 
from core.speech_to_text.stt_processor import SpeechToTextProcessor
from services.summary_service import SummaryService
from services.milvus_service import MilvusManager # 导入 MilvusManager (通用型)
from models.transcript_entry import TranscriptEntry # 导入 TranscriptEntry
from core.knowledge_engine.kg_builder import KnowledgeGraphBuilder # 导入 KnowledgeGraphBuilder
from backend.connection_manager import ConnectionManager # WebSocket 连接管理器

logger = logging.getLogger(__name__)

class MinuteGenerator:
    """
    智能会议助手，负责协调语音转文本、声纹识别、实时摘要、知识图谱更新等功能。
    【已重命名】为 MinuteGenerator，以更好地反映其核心职责。
    """
    def __init__(self, 
                 settings_obj: settings,
                 mongodb_manager: MongoDBManager, 
                 meeting_milvus_manager: MilvusManager, 
                 llm_model: LLMModel,
                 bge_model: BGEEmbeddingModel,
                 neo4j_service: Neo4jService, 
                 stt_processor: SpeechToTextProcessor,
                 voiceprint_service: VoiceprintService, 
                 summary_service: SummaryService,
                 connection_manager: ConnectionManager 
                ):
        logger.info("初始化 MinuteGenerator...")
        self.settings = settings_obj
        self.mongodb_manager = mongodb_manager 
        self.meeting_milvus_manager = meeting_milvus_manager 
        self.llm_model = llm_model
        self.bge_model = bge_model
        self.neo4j_service = neo4j_service
        self.stt_processor = stt_processor
        self.voiceprint_service = voiceprint_service 
        self.summary_service = summary_service
        self.connection_manager = connection_manager 

        self.realtime_transcript_buffer: List[TranscriptEntry] = [] # 实时转录缓冲区
        self.meeting_topics: Dict[str, Any] = {} # 存储会议话题
        self.current_meeting_id: Optional[str] = None # 当前活跃会议ID
        self.last_speaker_id: Optional[str] = None # 上一个发言人ID
        self.last_speaker_change_time: Optional[datetime] = None # 上次发言人切换时间

        # 实例化 KnowledgeGraphBuilder
        self.kg_builder = KnowledgeGraphBuilder(
            neo4j_service=self.neo4j_service,
            llm_model=self.llm_model,
            settings_obj=self.settings
        )
        logger.info("KnowledgeGraphBuilder 实例已创建。")

        logger.info("MinuteGenerator 初始化完成。")

    async def process_real_time_audio(self, audio_chunk: np.ndarray, sample_rate: int, client_id: str, meeting_id: str):
        """
        处理实时音频块。优先进行说话人分离和声纹识别，然后进行语音转文本。
        【已修改】现在将所有片段的转录和后续处理都放在一个循环中。
        Args:
            audio_chunk (np.ndarray): 实时音频数据块。
            sample_rate (int): 音频采样率。
            client_id (str): 发送音频的客户端 ID。
            meeting_id (str): 会议 ID。
        """
        self.current_meeting_id = meeting_id # 确保当前会议ID已设置
        
        try:
            # 1. 将音频块传递给 VoiceprintService 进行说话人分离和识别
            voice_segments_with_speakers = await self.voiceprint_service.process_realtime_audio(
                stt_processor=self.stt_processor,
                audio_chunk=audio_chunk, 
                sample_rate=sample_rate, 
                meeting_id=meeting_id
            )

            if not voice_segments_with_speakers:
                # logger.debug(f"VoiceprintService 未检测到有效语音片段或说话人 (客户端: {client_id}, 会议: {meeting_id})。")
                return # 没有可处理的片段，直接返回

            # 2. 遍历每个带有说话人信息的语音片段，进行 STT 和后续处理
            for segment_info in voice_segments_with_speakers:
                segment_audio = segment_info['audio']
                segment_sample_rate = segment_info['sample_rate']
                speaker_id = segment_info['user_id']
                speaker_name = segment_info['username']
                speaker_role = segment_info['role']
                transcript_text = segment_info.get('text', '').strip()

                if not transcript_text:
                    logger.debug(f"STT 转录结果为空 for segment (说话人: {speaker_name})。")
                    continue # 没有文本，跳过当前片段

                logger.info(f"STT 结果 (会议: {meeting_id}, 说话人: {speaker_name}): {transcript_text}")

                # 2.2. 存储转录到 MongoDB
                transcript_entry = TranscriptEntry(
                    meeting_id=meeting_id,
                    client_id=client_id,
                    user_id=speaker_id,
                    speaker_id=speaker_name,
                    role=speaker_role,
                    text=transcript_text,
                    timestamp=datetime.now(timezone.utc)
                )
                mongo_doc_id = await self.mongodb_manager.save_transcript_entry(transcript_entry) 
                logger.info(f"转录已保存到 MongoDB (会议: {meeting_id}, 发言人: {speaker_name}, MongoDB ID: {mongo_doc_id})。")

                # 2.3. 生成文本嵌入并存储到 Milvus
                if self.settings.ENABLE_STT and self.bge_model and self.bge_model.is_model_loaded():
                    text_embedding = await self.bge_model.get_embedding(transcript_text)
                    if text_embedding and mongo_doc_id:
                        milvus_data_entry = {
                            "embedding": text_embedding,
                            "mongo_id": str(mongo_doc_id)
                        }
                        try:
                            await self.meeting_milvus_manager.insert_data([milvus_data_entry])
                            logger.info(f"文本嵌入已插入 Milvus 集合 '{self.settings.MILVUS_MEETING_COLLECTION_NAME}' (MongoDB ID: {mongo_doc_id})。")
                        except Exception as e:
                            logger.error(f"插入文本嵌入到 Milvus 失败 (MongoDB ID: {mongo_doc_id}): {e}", exc_info=True)
                    else:
                        logger.warning(f"无法生成文本嵌入或 MongoDB ID 为空，跳过 Milvus 插入 (会议: {meeting_id}, 发言人: {speaker_name})。")
                else:
                    logger.debug("STT 未启用或 BGE 模型未加载，跳过文本嵌入生成和 Milvus 插入。")

                # 2.4. 广播实时转录结果给所有会议参与者
                await self._broadcast_realtime_transcript(transcript_entry)

        except Exception as e:
            logger.error(f"实时音频处理失败 (客户端: {client_id}, 会议: {meeting_id}): {e}", exc_info=True)

    async def _update_realtime_features(self, meeting_id: str, new_transcript: TranscriptEntry):
        """
        异步更新实时功能，如话题检测和知识图谱更新。
        【已修改】移除了实时摘要生成和广播的逻辑。
        """
        try:
            # 话题检测 (每隔一定数量的发言进行一次)
            if len(self.realtime_transcript_buffer) % self.settings.TOPIC_DETECTION_INTERVAL == 0:
                topic_context_text = " ".join([t.text for t in self.realtime_transcript_buffer[-self.settings.TOPIC_DETECTION_CONTEXT_LINES:] if t.text])
                if topic_context_text:
                    topics = await self.llm_model.extract_topics(topic_context_text)
                    if topics:
                        self.meeting_topics[meeting_id] = topics # 更新会议话题
                        await self._broadcast_meeting_topics(topics)

            # 知识图谱更新 (如果需要，可以将相关实体和关系提取并更新到 Neo4j)
            if self.settings.ENABLE_STT:
                # 这部分逻辑保持不变
                pass

        except Exception as e:
            logger.error(f"更新实时功能失败 (会议: {meeting_id}): {e}", exc_info=True)
    
    async def _broadcast_realtime_transcript(self, transcript_entry: TranscriptEntry):
        """
        将实时转录结果广播到所有连接的客户端，确保消息格式与前端匹配。
        """
        try:
            # 1. 检查连接管理器是否有效
            if not self.connection_manager:
                logger.error("ConnectionManager 未初始化，无法广播。")
                return

            # 根据用户ID获取用户信息，以便发送给前端
            user_info = None
            if self.mongodb_manager and transcript_entry.user_id:
                user_info = await self.mongodb_manager.get_user(transcript_entry.user_id)
            
            # 构造可序列化的消息体，确保字段名与前端完全匹配
            message = {
                "type": "transcript_update",  # <-- 将消息类型改为前端期望的
                "meetingId": transcript_entry.meeting_id,
                "clientId": transcript_entry.client_id,
                "userId": transcript_entry.user_id,
                "speakerId": transcript_entry.speaker_id, # <-- 确保字段名为 speakerId
                "role": transcript_entry.role,
                "text": transcript_entry.text, # <-- 确保字段名为 text
                "timestamp": transcript_entry.timestamp.isoformat(),
                "isFinal": True # 我们假设实时转录都是最终的
            }
            
            logger.debug(f"准备广播消息: {message}")
            
            # 执行广播操作
            await self.connection_manager.send_message_to_meeting(message, transcript_entry.meeting_id) 
            logger.info(f"✅ 已成功广播实时转录到会议 {transcript_entry.meeting_id}。")
        
        except Exception as e:
            logger.error(f"广播实时转录消息失败: {e}", exc_info=True)

    async def _broadcast_meeting_participants(self, meeting_id: str):
        """
        获取并广播当前会议的所有在线参与者列表。
        """
        if self.connection_manager:
            participants = await self.mongodb_manager.get_users_in_meeting(
                meeting_id,
                self.connection_manager.get_active_clients_in_meeting(meeting_id)
            )
            message = {
                "type": "participant_list_update",
                "meetingId": meeting_id,
                "participants": [p.model_dump() for p in participants],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.connection_manager.send_message_to_meeting(message, meeting_id)
            logger.info(f"广播会议 {meeting_id} 的参与者列表。")
        else:
            logger.error("ConnectionManager 未设置，无法广播会议参与者列表。")

    async def _broadcast_meeting_topics(self, topics: List[str]):
        """
        向所有连接到当前会议的客户端广播会议话题。
        """
        if self.connection_manager and self.current_meeting_id:
            message = {
                "type": "meeting_topics",
                "meetingId": self.current_meeting_id,
                "topics": topics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.connection_manager.send_message_to_meeting(message, self.current_meeting_id)
            logger.debug(f"广播会议话题到会议 {self.current_meeting_id}。")
        else:
            logger.error("ConnectionManager 或 current_meeting_id 未设置，无法广播会议话题。")

    async def generate_meeting_summary_by_id(self, meeting_id: str) -> Dict[str, Any]:
        """
        【新增方法】
        根据会议ID，从数据库获取所有转录内容，并生成最终会议摘要。
        此方法用于响应前端按需生成摘要的请求。
        """
        logger.info(f"正在为会议 {meeting_id} 按需生成摘要...")

        full_transcript_entries = await self.mongodb_manager.get_all_transcripts_for_meeting(meeting_id)
        if not full_transcript_entries:
            raise ValueError("找不到会议的任何转录内容。")

        full_text_for_summary = " ".join([t.text for t in full_transcript_entries if t.text])
        if not full_text_for_summary.strip():
            raise ValueError("会议内容不足以生成摘要。")

        # 2. 生成最终摘要
        final_summary = await self.summary_service.generate_summary(
            full_text_for_summary,
            max_length=self.settings.BART_FINAL_SUMMARY_MAX_LENGTH,
            min_length=self.settings.BART_FINAL_SUMMARY_MIN_LENGTH
        )
        logger.info("最终会议摘要生成完成。")

        # 3. 提取行动项和决策 (使用 LLM)
        action_items = await self.llm_model.extract_action_items(full_text_for_summary)
        decisions = await self.llm_model.extract_decisions(full_text_for_summary)
        logger.info("行动项和决策提取完成。")

        return {
            "summary": final_summary,
            "actions": action_items,
            "decisions": decisions
        }


    async def process_meeting(self, audio_data: np.ndarray, sample_rate: int, transcript: str) -> Dict[str, Any]:
        """
        处理整个会议的音频和转录。
        方法现在是一个历史遗留或文件处理方法，其逻辑应与按需摘要生成分离。
        """
        logger.info("开始处理整个会议音频和转录...")
        
      

        temp_audio_path = None
        try:
            temp_audio_path = f"temp_meeting_audio_{uuid.uuid4()}.wav"
            sf.write(temp_audio_path, audio_data, sample_rate)
            logger.info(f"会议音频已保存到临时文件: {temp_audio_path}")

            diarization_results = await self.voiceprint_service.process_audio_for_diarization(temp_audio_path)
            logger.info(f"说话人分离完成，共 {len(diarization_results)} 个片段。")

            identified_speakers_summary = {}
            for entry in diarization_results:
                speaker_label = entry.get('speaker', 'Unknown')
                if speaker_label not in identified_speakers_summary:
                    identified_speakers_summary[speaker_label] = {"duration": 0.0, "segments": []}
                identified_speakers_summary[speaker_label]["duration"] += (entry['end'] - entry['start'])
                identified_speakers_summary[speaker_label]["segments"].append(f"[{entry['start']:.1f}-{entry['end']:.1f}s]")
            
            full_text_for_summary = transcript
            

            final_summary = await self.summary_service.generate_summary(
                full_text_for_summary,
                max_length=self.settings.BART_FINAL_SUMMARY_MAX_LENGTH,
                min_length=self.settings.BART_FINAL_SUMMARY_MIN_LENGTH
            )
            logger.info("最终会议摘要生成完成。")

            action_items = await self.llm_model.extract_action_items(full_text_for_summary)
            decisions = await self.llm_model.extract_decisions(full_text_for_summary)
            logger.info("行动项和决策提取完成。")

            return {
                "summary": final_summary,
                "actions": action_items,
                "decisions": decisions,
                "identified_speakers": identified_speakers_summary
            }
        except Exception as e:
            logger.error(f"处理整个会议失败: {e}", exc_info=True)
            raise
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logger.info(f"已删除临时会议音频文件: {temp_audio_path}")

    async def answer_question(self, meeting_id: str, question: str) -> str:
        """
        根据会议转录回答问题。
        现在使用 async for 循环来正确处理 llm_model.answer_question 返回的异步生成器。
        Args:
            meeting_id (str): 会议 ID。
            question (str): 用户提出的问题。
        Returns:
            str: 问题的完整答案。
        """
        logger.info(f"正在为会议 {meeting_id} 回答问题: '{question}'")
        
        full_transcript_entries = await self.mongodb_manager.get_all_transcripts_for_meeting(meeting_id) 
        if not full_transcript_entries:
            return "对不起，我没有找到这个会议的任何转录内容。"

        context_lines = self.settings.QA_CONTEXT_LINES
        context_texts = [entry.text for entry in full_transcript_entries[-context_lines:]]
        context = " ".join(context_texts)

        if not context.strip():
            return "对不起，会议内容不足以回答您的问题。"

        # 正确处理异步生成器
        answer_fragments = []
        async for fragment in self.llm_model.answer_question(context, question):
            answer_fragments.append(fragment)
        
        answer = "".join(answer_fragments)
        return answer
    # 使用 SummaryService 生成会议摘要
    async def generate_summary(self, transcript_segments: List[Dict[str, Any]], max_length: int = 150, min_length: int = 40, num_beams: int = 4) -> dict[str, str]:
        """
        使用 SummaryService 生成会议纪要的摘要。
        """
        logger.info("正在生成会议摘要...")
    
        cleaned_segments = []
        for seg in transcript_segments:
            if isinstance(seg, dict) and 'text' in seg and seg['text']:
                cleaned_segments.append(seg)
            else:
                logger.warning(f"检测到无效的转录段落，已跳过。段落内容: {seg}")
    
        if not cleaned_segments:
            logger.warning("转录内容为空，无法生成摘要。")
            return {"detail": "会议内容不足以生成摘要。"}
        
        full_transcript = " ".join([seg['text'] for seg in cleaned_segments])
    
        try:
            # 正确的做法：调用 self.summary_service 的方法
            summary_result = await self.summary_service.generate_summary(
                text=full_transcript,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams
            )
            
            # 检查返回的摘要结果是否包含错误信息
            if "detail" in summary_result:
                logger.error(f"Summary generation failed from SummaryService: {summary_result['detail']}")
                return {"detail": summary_result['detail']}
    
            summary_text = summary_result.get("summary", "")
    
            # 检查生成的摘要质量
            if not summary_text or len(summary_text) < 10 or "..." in summary_text:
                logger.warning(f"Summary generation incomplete or empty, input length: {len(full_transcript)}, summary: {summary_text}")
                return {"detail": "摘要生成不完整或为空，请调整参数或检查模型状态。"}
            
            logger.info(f"会议摘要生成成功。")
            return {"summary": summary_text}
        
        except Exception as e:
            logger.error(f"调用 SummaryService 失败: {str(e)}", exc_info=True)
            return {"detail": f"处理失败: {str(e)}"}


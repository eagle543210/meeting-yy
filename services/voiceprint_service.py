# services\voiceprint_service.py

import os
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import asyncio
import uuid
from dotenv import load_dotenv 
import time
from datetime import datetime

# 导入配置
from config.settings import settings
from services.milvus_service import MilvusManager
from services.mongodb_manager import MongoDBManager
from models import User, UserRole
from core.speech_to_text.stt_processor import SpeechToTextProcessor 

load_dotenv()
logger = logging.getLogger(__name__)

# 尝试导入 pyannote.audio
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.core.model import Model
    from pyannote.core import Segment, Annotation
    from pyannote.audio import Audio
    import torchaudio
    logger.info("pyannote.audio 和 torchaudio 导入成功。")
except ImportError:
    logger.warning("无法导入 pyannote.audio 或 torchaudio。声纹识别和说话人分离功能将受限。")
    Pipeline = None
    Model = None
    Audio = None
    torchaudio = None

class VoiceprintService:
    """
    VoiceprintService 负责声纹的注册、识别和说话人分离。
    此版本实现了延迟注册策略：
    - 短音频片段（不足以提取可靠声纹）将被临时存储。
    - 当一个足够长的音频片段出现时，将尝试注册新声纹，并将之前的临时片段合并到新用户ID下。
    """
    def __init__(self, settings_obj: settings, voice_milvus_manager: MilvusManager, mongodb_manager: MongoDBManager):
        logger.info("初始化 VoiceprintService...")
        self.settings = settings_obj

        self.device_str = "cuda" if self.settings.USE_CUDA and torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        logger.info(f"检测到的设备: {self.device_str.upper()}")
        self.min_segment_duration = getattr(self.settings, 'MIN_AUDIO_SEGMENT_DURATION_S', 0.5)
        # 这里的 MIN_AUDIO_SAMPLES 和 VOICE_EMBEDDING_MIN_DURATION 都用于定义最小声纹时长
        self.MIN_AUDIO_SAMPLES = int(self.settings.MIN_SPEECH_SEGMENT_DURATION * self.settings.VOICE_SAMPLE_RATE)
        logger.info(f"声纹嵌入最小音频长度设置为 {self.MIN_AUDIO_SAMPLES} 采样点 ({self.MIN_AUDIO_SAMPLES / self.settings.VOICE_SAMPLE_RATE:.1f} 秒)。")

        self.diarization_pipeline: Optional[Pipeline] = None
        self.embedding_model: Optional[Model] = None 
        self.audio_processor: Optional[Audio] = None
        self._model_loaded: bool = False
        self.voice_milvus_manager = voice_milvus_manager
        self.mongodb_manager = mongodb_manager
        self.registered_voiceprints_cache: Dict[str, Dict[str, Any]] = {}
        
        self.realtime_buffers: Dict[str, List[np.ndarray]] = {} 
        self.buffer_start_time: Dict[str, float] = {} 
        
        self.min_speech_off_duration = self.settings.MIN_SPEECH_DURATION_OFF

        # ---用于存储未注册声纹的临时数据 ---
        # 键为 pyannote 的临时说话人标签，值为包含多个短音频片段和文本的列表
        self.pending_speaker_data: Dict[str, List[Dict[str, Any]]] = {}
        # 最小的语音片段长度（秒），用于注册新声纹，从设置中获取
        self.min_voiceprint_duration = self.settings.VOICE_EMBEDDING_MIN_DURATION

        logger.info("VoiceprintService 初始化完成，模型待异步加载。")

    async def load_model(self):
        """
        异步加载说话人分离和嵌入模型。
        """
        if self._model_loaded:
            logger.info("VoiceprintService 模型已加载，跳过重复加载。")
            return
        
        try:
            os.environ["HF_HUB_OFFLINE"] = self.settings.HF_HUB_OFFLINE
            
            hf_token = None
            if self.settings.HF_TOKEN:
                if isinstance(self.settings.HF_TOKEN, str):
                    hf_token = self.settings.HF_TOKEN
                elif hasattr(self.settings.HF_TOKEN, 'get_secret_value'):
                    hf_token = self.settings.HF_TOKEN.get_secret_value()
            
            if not hf_token and self.settings.HF_HUB_OFFLINE == "0":
                logger.warning("HF_TOKEN 未设置，在线模式下可能无法下载 pyannote.audio 模型。")

            if Pipeline:
                logger.info(f"尝试加载说话人分离模型: '{self.settings.PYANNOTE_DIARIZATION_MODEL}'...")
                self.diarization_pipeline = await asyncio.to_thread(
                    Pipeline.from_pretrained,
                    self.settings.PYANNOTE_DIARIZATION_MODEL,
                    use_auth_token=hf_token,
                )
                
                def all_speech_vad(file):
                    duration = file['waveform'].shape[1] / file['sample_rate']
                    annotation = Annotation()
                    annotation[Segment(0, duration)] = 'SPEECH'
                    return annotation

                self.diarization_pipeline.vad = all_speech_vad
                self.diarization_pipeline.to(self.device)
                logger.info("🎉 说话人分离模型已成功加载。")
            else:
                logger.warning("pyannote.audio.Pipeline 未导入，说话人分离模型无法加载。")
                raise RuntimeError("pyannote.audio.Pipeline 未导入，服务无法启动。")

            if Model:
                logger.info(f"尝试加载说话人嵌入模型: '{self.settings.PYANNOTE_EMBEDDING_MODEL}'...")
                self.embedding_model = await asyncio.to_thread(
                    Model.from_pretrained,
                    self.settings.PYANNOTE_EMBEDDING_MODEL,
                    use_auth_token=hf_token,
                    strict=False,
                )
                self.embedding_model.to(self.device)
                logger.info("🎉 说话人嵌入模型已成功加载。")
            else:
                logger.warning("pyannote.audio.core.model.Model 未导入，说话人嵌入模型无法加载。")
                raise RuntimeError("pyannote.audio.core.model.Model 未导入，服务无法启动。")

            if Audio:
                self.audio_processor = Audio(sample_rate=self.settings.VOICE_SAMPLE_RATE)
                logger.info(f"Audio 处理器采样率设置为: {self.audio_processor.sample_rate} Hz。")
            else:
                logger.warning("pyannote.audio.Audio 未导入，音频处理器无法初始化。")
                raise RuntimeError("pyannote.audio.Audio 未导入，服务无法启动。")

            await self._load_registered_voiceprints_from_milvus()

            self._model_loaded = True
            logger.info("VoiceprintService 模型加载完成。")
        except Exception as e:
            logger.critical(f"❌ 错误：VoiceprintService 模型加载失败！")
            logger.critical(f"错误信息: {e}", exc_info=True)
            self.diarization_pipeline = None
            self.embedding_model = None
            self.audio_processor = None
            self._model_loaded = False
            raise RuntimeError(f"VoiceprintService 初始化失败: {str(e)}") from e

    def is_model_loaded(self) -> bool:
        """
        检查所有模型是否已加载。
        """
        return self._model_loaded

    async def _load_registered_voiceprints_from_milvus(self):
        """
        从 Milvus 加载所有已注册的声纹元数据到本地缓存。
        缓存结构为 { user_id: { "embedding": np.ndarray, "username": str, "role": str } }
        """
        logger.info("正在从 Milvus 加载已注册声纹到本地缓存...")
        if not self.voice_milvus_manager or not self.voice_milvus_manager.is_connected:
            logger.error("MilvusManager 未连接或未初始化，无法加载已注册声纹。")
            self.registered_voiceprints_cache = {}
            return

        try:
            milvus_data = await self.voice_milvus_manager.get_all_data(output_fields=["id", "user_name", "role", "embedding"])
            
            self.registered_voiceprints_cache = {}
            for entry in milvus_data:
                user_id = entry.get("id")
                embedding = entry.get("embedding")
                user_name = entry.get("user_name")
                role = entry.get("role")

                if user_id and embedding and user_name and role:
                    self.registered_voiceprints_cache[user_id] = {
                        "embedding": np.array(embedding, dtype=np.float32),
                        "username": user_name,
                        "role": role
                    }
                    logger.debug(f"已加载声纹: {user_name} ({user_id})")
                else:
                    logger.warning(f"从 Milvus 加载声纹时发现不完整数据: {entry}")

            logger.info(f"成功从 Milvus 加载 {len(self.registered_voiceprints_cache)} 条声纹到缓存。")
        except Exception as e:
            logger.error(f"从 Milvus 加载已注册声纹到缓存失败: {e}", exc_info=True)
            self.registered_voiceprints_cache = {}

    async def _get_embedding(self, audio_data: np.ndarray, sample_rate: int) -> Optional[List[float]]:
        """
        从音频数据中提取单个说话人的声纹嵌入。
        """
        if not self.embedding_model or not self.audio_processor or not torchaudio:
            logger.error("声纹嵌入模型、音频处理器或 torchaudio 未加载，无法提取声纹。")
            return None

        # 最小采样点数，用于提取声纹
        min_samples_for_embedding = int(self.settings.VOICE_EMBEDDING_MIN_DURATION * self.settings.VOICE_SAMPLE_RATE)

        if audio_data.shape[-1] < min_samples_for_embedding:
            logger.warning(f"音频片段太短 ({audio_data.shape[-1]} 采样点)，不足以提取声纹。最小要求: {min_samples_for_embedding} 采样点。")
            return None
        
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            logger.error("输入音频数据包含 NaN 或 Inf 值，无法进行声纹提取。")
            return None

        try:
            waveform = torch.from_numpy(audio_data).float().to(self.device)
            
            if sample_rate != self.settings.VOICE_SAMPLE_RATE:
                logger.debug(f"重采样音频从 {sample_rate} Hz 到 {self.settings.VOICE_SAMPLE_RATE} Hz。")
                resampler = torchaudio.transforms.Resample(sample_rate, self.settings.VOICE_SAMPLE_RATE).to(self.device)
                waveform = resampler(waveform)
            
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                if waveform.shape[0] == 1:
                    waveform = waveform.unsqueeze(1)
                elif waveform.shape[1] == 1:
                    waveform = waveform.permute(1, 0).unsqueeze(0)
                else:
                    waveform = waveform.unsqueeze(0)
                
            with torch.no_grad():
                embedding = await asyncio.to_thread(self.embedding_model, waveform)
            
            return embedding.cpu().detach().numpy().squeeze().tolist()

        except Exception as e:
            logger.error(f"提取声纹嵌入失败: {e}", exc_info=True)
            return None

    async def register_voice(self, audio_data: np.ndarray, sample_rate: int, user_id: str, username: str, role: str) -> Dict[str, Any]:
        """
        注册用户的声纹。
        """
        logger.info(f"VoiceprintService: 尝试注册声纹 for user_id: {user_id}, username: {username}, role: {role}")
        if not self.voice_milvus_manager or not self.voice_milvus_manager.is_connected:
            raise RuntimeError("MilvusManager 未初始化或未连接。无法注册声纹。")
        if not self.mongodb_manager:
            raise RuntimeError("MongoDBManager 未初始化。无法注册声纹。")

        embedding = await self._get_embedding(audio_data, sample_rate)
        if embedding is None:
            raise ValueError("无法从提供的音频生成声纹嵌入。请确保音频质量和时长符合要求。")

        try:
            milvus_data_entry = {
                "id": user_id,
                "user_name": username,
                "role": role,
                "embedding": embedding
            }
            
            pks = await self.voice_milvus_manager.insert_data([milvus_data_entry])
            
            if pks:
                self.registered_voiceprints_cache[user_id] = {
                    "embedding": np.array(embedding, dtype=np.float32),
                    "username": username,
                    "role": role
                }
                logger.info(f"声纹 for user_id: {user_id} 已成功注册到 Milvus 并缓存。")

                from models import User
                user_obj = User(user_id=user_id, username=username, role=UserRole(role.upper()))
                await self.mongodb_manager.add_or_update_user(user_obj)
                logger.info(f"用户 '{username}' (ID: {user_id}) 的元数据已保存/更新到 MongoDB。")

                return {"status": "registered", "user_id": user_id, "is_new_user": True, "message": "声纹注册成功"}
            else:
                raise RuntimeError("Milvus 插入操作未返回 ID。")
        except Exception as e:
            logger.error(f"注册声纹失败 for user_id: {user_id}: {e}", exc_info=True)
            raise RuntimeError(f"声纹注册失败: {str(e)}")

    async def identify_speaker(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        识别音频中的说话人。
        """
        logger.debug("VoiceprintService: 尝试识别说话人...")
        
        default_unknown_user = {"user_id": None, "username": "未知用户", "role": UserRole.UNKNOWN.value, "is_known": False, "confidence": 0}

        if not self.voice_milvus_manager or not self.voice_milvus_manager.is_connected:
            logger.error("MilvusManager 未初始化或未连接。无法识别说话人。")
            return default_unknown_user
        
        if not self.registered_voiceprints_cache:
            logger.warning("没有已注册的声纹，无法进行识别。将返回未知用户。")
            return default_unknown_user

        query_embedding = await self._get_embedding(audio_data, sample_rate)
        if query_embedding is None:
            logger.error("无法从查询音频中提取嵌入向量。将返回未知用户。")
            return default_unknown_user

        try:
            search_results = await self.voice_milvus_manager.search_data(
                query_vectors=[query_embedding], 
                top_k=1,
                output_fields=["user_name", "role"]
            )

            if search_results and search_results[0]: # 确保有结果且第一个结果列表不为空
                # Milvus search_data 返回一个列表，其中包含一个或多个 HybridHits 对象。
                # 我们需要获取第一个查询结果（search_results[0]），再获取第一个匹配的 Hit 对象。
                best_match = search_results[0][0]
                
                # ---直接通过属性访问 Milvus Hit 对象的数据 ---
                user_id = best_match.id
                distance = best_match.distance
                
                # 访问 entity 属性来获取 output_fields 中的额外数据
                username = best_match.entity.get("user_name", f"未知用户_{user_id[:6] if user_id else 'N/A'}")
                role = best_match.entity.get("role", UserRole.GUEST.value)
                
                if distance is not None and distance <= self.settings.VOICEPRINT_SIMILARITY_THRESHOLD:
                    confidence = 1.0 - (distance / self.settings.VOICEPRINT_SIMILARITY_THRESHOLD)
                    confidence = max(0.0, min(1.0, confidence))
                    confidence_percent = int(confidence * 100)

                    logger.info(f"识别到说话人: {username} (ID: {user_id}), 距离: {distance:.4f}, 置信度: {confidence_percent}%)")
                    return {
                        "user_id": user_id,
                        "username": username,
                        "role": role,
                        "confidence": confidence_percent,
                        "is_known": True
                    }
                else:
                    logger.info(f"未找到足够相似的声纹 (最佳距离: {distance:.4f}, 阈值: {self.settings.VOICEPRINT_SIMILARITY_THRESHOLD})。将返回未知用户。")
                    return default_unknown_user
            else:
                logger.info("未在 Milvus 中找到匹配的声纹。将返回未知用户。")
                return default_unknown_user
        except Exception as e:
            logger.error(f"识别说话人失败: {e}", exc_info=True)
            return {"user_id": "error", "username": "识别错误", "role": UserRole.ERROR.value, "is_known": False, "confidence": 0}


    # 这个方法在新逻辑中被 _process_sliding_window 替代，以实现更精细的控制
    # async def _auto_register_if_needed(...) -> Dict[str, Any]:
    #     ...

    async def _process_sliding_window(self, meeting_id: str, sample_rate: int, stt_processor: SpeechToTextProcessor) -> List[Dict[str, Any]]:
        """
        核心方法：处理累积的音频缓冲区。
        此方法只处理缓冲区的最新部分（即滑动窗口），并返回结果。
        此版本包含了延迟注册策略。
        """
        start_time = time.time()
        
        if meeting_id not in self.realtime_buffers or not self.realtime_buffers[meeting_id]:
            logger.warning(f"会议 {meeting_id} 缓冲区为空或不存在，无法处理。")
            self.buffer_start_time.pop(meeting_id, None)
            return []
            
        full_buffered_audio = np.concatenate(self.realtime_buffers[meeting_id])
        
        if full_buffered_audio.size == 0:
            logger.warning(f"会议 {meeting_id} 缓冲区为空，无法处理。")
            self.buffer_start_time.pop(meeting_id, None)
            return []
        
        window_duration = getattr(self.settings, 'REALTIME_SLIDING_WINDOW_S', 5.0)
        
        start_offset_in_samples = max(0, full_buffered_audio.shape[0] - int(window_duration * sample_rate))
        processed_audio = full_buffered_audio[start_offset_in_samples:]
        
        if processed_audio.ndim > 1:
            processed_audio = processed_audio.squeeze()
        if processed_audio.dtype != np.float32:
            processed_audio = processed_audio.astype(np.float32)
        
        waveform_tensor = torch.from_numpy(processed_audio).float().to(self.device).unsqueeze(0)
        audio_duration = processed_audio.shape[0] / sample_rate
        
        logger.debug(f"正在将 {audio_duration:.2f} 秒的音频数据送入 pyannote.audio 进行说话人分离。")
        
        diarization = await asyncio.to_thread(
            self.diarization_pipeline,
            {"waveform": waveform_tensor, "sample_rate": sample_rate}
        )

        results = []
        
        # 使用配置文件中已有的 PYANNOTE_MIN_SPEECH_DURATION_S 设置
        min_speech_duration = self.settings.PYANNOTE_MIN_SPEECH_DURATION_S

        if not diarization or len(diarization) == 0:
            logger.warning(f"Pyannote Diarization 未检测到任何语音活动，强制创建一个默认语音片段。")
            diarization = Annotation()
            diarization[Segment(0, audio_duration)] = 'SPEAKER_00'

        for segment, track, speaker_label in diarization.itertracks(yield_label=True):
            segment_start = segment.start
            segment_end = segment.end
            
            segment_duration = segment_end - segment_start
            
            if segment_duration < min_speech_duration:
                logger.debug(f"跳过空或过短的语音片段 ({segment_duration:.2f}s)。")
                continue

            start_idx = int(segment_start * sample_rate)
            end_idx = int(segment_end * sample_rate)
            segment_audio = processed_audio[start_idx:end_idx]
            
            global_start_time = self.buffer_start_time.get(meeting_id, 0.0) + start_offset_in_samples / sample_rate + segment_start
            global_end_time = self.buffer_start_time.get(meeting_id, 0.0) + start_offset_in_samples / sample_rate + segment_end

            transcription_result = await stt_processor.transcribe_audio(segment_audio, sample_rate)
            transcribed_text = transcription_result.get("text", "") if transcription_result else ""
            confidence = transcription_result.get("confidence", 0.0) if transcription_result else 0.0
            
            # --- 延迟注册和合并逻辑的核心 ---
            if segment_duration < self.min_voiceprint_duration:
                # 片段过短，存入临时缓冲区
                logger.info(f"音频片段过短 ({segment_duration:.2f}s)，存入临时缓冲区。Pyannote标签: {speaker_label}, 文本: '{transcribed_text}'")
                
                if speaker_label not in self.pending_speaker_data:
                    self.pending_speaker_data[speaker_label] = []
                self.pending_speaker_data[speaker_label].append({
                    "audio": segment_audio,
                    "text": transcribed_text,
                    "start_time": global_start_time,
                    "end_time": global_end_time,
                })
                continue
            else:
                # 片段足够长，尝试识别或注册
                identified_user_info = await self.identify_speaker(segment_audio, sample_rate)
                
                final_user_info = identified_user_info
                
                if not identified_user_info.get("is_known", False):
                    # 未知用户，注册新声纹并合并缓冲区数据
                    new_user_id = str(uuid.uuid4())
                    new_username = f"未知用户_{new_user_id[:8]}"
                    new_role = UserRole.GUEST.value
                    
                    registration_result = await self.register_voice(
                        audio_data=segment_audio,
                        sample_rate=sample_rate,
                        user_id=new_user_id,
                        username=new_username,
                        role=new_role
                    )
                    
                    if registration_result.get("status") == "registered":
                        logger.info(f"新用户 '{new_username}' (ID: {new_user_id}) 注册成功。")
                        final_user_info = {
                            "user_id": new_user_id,
                            "username": new_username,
                            "role": new_role,
                            "confidence": 100,
                            "is_known": True
                        }
                    else:
                        logger.error(f"自动注册新用户失败: {new_username}")
                        continue # 注册失败，跳过此片段
                
                # 检查并合并该说话人之前在缓冲区中的数据
                merged_data = self._check_and_merge_pending_data(speaker_label)
                if merged_data:
                    # 合并音频和文本
                    full_audio = np.concatenate([d['audio'] for d in merged_data] + [segment_audio])
                    full_text = " ".join([d['text'] for d in merged_data] + [transcribed_text])
                    # 更新时间戳
                    global_start_time = merged_data[0]['start_time']
                    global_end_time = global_end_time
                    logger.info(f"已合并来自 {speaker_label} 的 {len(merged_data)} 个临时片段。")
                    # 使用合并后的完整数据进行最终输出
                    transcribed_text = full_text
                    segment_audio = full_audio

            if not transcribed_text:
                logger.warning(f"STT 转录未检测到有效文本。说话人: {final_user_info.get('username')}, 时间: {global_start_time:.2f}-{global_end_time:.2f}s")
            else:
                logger.info(f"STT 转录成功。说话人: {final_user_info.get('username')}, 时间: {global_start_time:.2f}-{global_end_time:.2f}s, 文本: '{transcribed_text}'")

            results.append({
                "audio": segment_audio,
                "sample_rate": sample_rate,
                "start_time": global_start_time,
                "end_time": global_end_time,
                "temp_speaker_id": speaker_label,
                "user_id": final_user_info.get("user_id"),
                "username": final_user_info.get("username"),
                "role": final_user_info.get("role"),
                "is_new_user": not identified_user_info.get("is_known", False),
                "text": transcribed_text,
                "confidence": confidence,
            })
            
        # 处理完后，将缓冲区中已处理的部分移除，实现“滑动”
        processed_until_samples = full_buffered_audio.shape[0]
        self.realtime_buffers[meeting_id] = [full_buffered_audio[processed_until_samples:]]
        self.buffer_start_time[meeting_id] += processed_until_samples / sample_rate

        end_time = time.time()
        logger.debug(f"_process_sliding_window 方法执行耗时: {(end_time - start_time):.4f}s") 
        
        return results

    def _check_and_merge_pending_data(self, speaker_label: str) -> List[Dict[str, Any]]:
        """
        检查是否有属于此说话人标签的临时数据，并进行合并。
        返回合并后的数据列表，并清空缓冲区。
        """
        if speaker_label in self.pending_speaker_data:
            merged_data = self.pending_speaker_data.pop(speaker_label)
            return merged_data
        return []

    async def process_realtime_audio_for_segments(
        self, audio_chunk: np.ndarray, sample_rate: int, meeting_id: str, stt_processor: SpeechToTextProcessor
    ) -> List[Dict[str, Any]]:
        """
        使用“滑动窗口”模式。
        此方法负责将新音频块添加到缓冲区，并触发处理。
        """
        if not self._model_loaded or self.diarization_pipeline is None:
            logger.error("Pyannote Diarization 模型未加载，无法执行实时音频处理。")
            return []

        # 确保会议的音频缓冲区已初始化
        if meeting_id not in self.realtime_buffers:
            self.realtime_buffers[meeting_id] = []
            # 使用当前时间作为缓冲区的起始时间戳
            self.buffer_start_time[meeting_id] = datetime.now().timestamp()
            
        # 将新的音频块添加到缓冲区
        self.realtime_buffers[meeting_id].append(audio_chunk)
        
        results = []
        
        # 拼接缓冲区中的所有音频块
        full_buffered_audio = np.concatenate(self.realtime_buffers[meeting_id])
        current_duration = full_buffered_audio.shape[0] / sample_rate
        
        # 获取滑动窗口的配置
        window_duration = getattr(self.settings, 'REALTIME_SLIDING_WINDOW_S', 5.0)

        # 如果累积的音频时长超过了滑动窗口的长度，就触发处理
        if current_duration >= window_duration:
            try:
                # 触发滑动窗口处理
                results = await self._process_sliding_window(
                    meeting_id, sample_rate, stt_processor
                )
            except Exception as e:
                logger.error(f"在处理会议 {meeting_id} 的实时音频时发生错误：{e}", exc_info=True)
                # 发生错误时，清空缓冲区以避免无限循环
                self.realtime_buffers.pop(meeting_id, None)
                self.buffer_start_time.pop(meeting_id, None)
        
        return results

    async def process_audio_for_diarization(self, audio_file_path: str) -> List[Dict[str, Any]]:
        """
        对给定的音频文件执行说话人分离，并返回结构化的结果。
        """
        if not self.diarization_pipeline:
            raise RuntimeError("说话人分离模型未加载，无法执行说话人分离。")

        if not os.path.exists(audio_file_path):
            logger.error(f"音频文件不存在: {audio_file_path}")
            raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")

        logger.info(f"正在对音频文件 '{audio_file_path}' 进行说话人分离...")
        try:
            diarization = await asyncio.to_thread(self.diarization_pipeline, audio_file_path)
            
            structured_results = []
            for segment, _, speaker_label in diarization.itertracks(yield_label=True):
                structured_results.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker_label
                })
            logger.info(f"说话人分离完成。检测到 {len(set(r['speaker'] for r in structured_results))} 个说话人。")
            return structured_results
        except Exception as e:
            logger.error(f"对音频文件 '{audio_file_path}' 进行说话人分离失败！", exc_info=True)
            raise Exception(f"说话人分离失败: {e}") from e

    async def close(self):
        """
        关闭 VoiceprintService，释放模型资源。
        """
        logger.info("Closing VoiceprintService...")
        self.diarization_pipeline = None
        self.embedding_model = None
        self.audio_processor = None
        self._model_loaded = False
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache for VoiceprintService: {e}")
        logger.info("VoiceprintService closed.")

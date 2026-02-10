# M:\meeting\services\voiceprint_service.py

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

        # --- 新增：用于VAD驱动的实时音频缓冲 ---
        # 键为 meeting_id，值为累积的音频数据块列表
        self.speech_buffer: Dict[str, List[np.ndarray]] = {}
        # 键为 meeting_id，值为最后一次检测到语音活动的时间戳
        self.last_speech_timestamp: Dict[str, float] = {}

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
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        self.settings.PYANNOTE_DIARIZATION_MODEL,
                        use_auth_token=hf_token
                    )
                    if self.diarization_pipeline:
                        self.diarization_pipeline.to(self.device)
                        logger.info("✅ Pyannote Diarization Pipeline 加载成功。")
                except Exception as e:
                    logger.error(f"加载 Pyannote Diarization Pipeline 失败: {e}")

            # --- 新增: 加载 Silero VAD 模型 ---
            try:
                vad_model_path = self.settings.VAD_MODEL_PATH
                if vad_model_path.exists():
                    self.vad_model, self.vad_utils = torch.hub.load(
                        repo_or_dir=str(self.settings.BASE_DIR / "models" / "silero_vad"),
                        model='silero_vad',
                        source='local',
                        onnx=False
                    )
                    self.vad_model.to(self.device)
                    logger.info(f"✅ Silero VAD 模型加载成功: {vad_model_path}")
                else:
                    logger.warning(f"⚠️ VAD 模型文件不存在: {vad_model_path}，将回退到基础能量检测。")
                    self.vad_model = None
            except Exception as e:
                logger.error(f"加载 Silero VAD 失败: {e}")
                self.vad_model = None

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
                
                # --- 修复点：直接通过属性访问 Milvus Hit 对象的数据 ---
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


    async def process_realtime_audio(self, audio_chunk: np.ndarray, sample_rate: int, meeting_id: str, stt_processor: SpeechToTextProcessor) -> List[Dict[str, Any]]:
        """
        [优化] 使用 Silero VAD 驱动的实时音频处理。
        支持：
        1. 静音触发 (VAD_PAUSE_DURATION_S)
        2. 超时强制触发 (MAX_UTTERANCE_DURATION_S)
        """
        if not self._model_loaded:
            return []

        current_time = time.time()
        
        # 初始化会议状态
        if meeting_id not in self.speech_buffer:
            self.speech_buffer[meeting_id] = []
            self.last_speech_timestamp[meeting_id] = current_time
            self.buffer_start_time[meeting_id] = current_time

        # --- 1. 使用 Silero VAD 进行检测 (如果不可用则回退) ---
        has_speech = False
        if self.vad_model:
            try:
                audio_tensor = torch.from_numpy(audio_chunk).float().to(self.device)
                # Silero VAD 期望 [batch, samples] 或 [samples]
                # 这里假设 chunk 已经足够长 (如 32ms+)
                speech_prob = self.vad_model(audio_tensor, sample_rate).item()
                has_speech = speech_prob > self.settings.VAD_SPEECH_THRESHOLD
            except Exception as e:
                logger.error(f"Silero VAD 处理失败: {e}")
                has_speech = np.abs(audio_chunk).max() > self.settings.AUDIO_ENERGY_THRESHOLD
        else:
            # 回退到能量级别检测
            has_speech = np.abs(audio_chunk).max() > self.settings.AUDIO_ENERGY_THRESHOLD

        # --- 2. 更新缓冲区 logic ---
        if has_speech:
            self.speech_buffer[meeting_id].append(audio_chunk)
            self.last_speech_timestamp[meeting_id] = current_time
            # 如果是刚开始说话，记录起始时间
            if len(self.speech_buffer[meeting_id]) == 1:
                self.buffer_start_time[meeting_id] = current_time
        
        # --- 3. 检查触发条件 ---
        should_process = False
        reason = ""
        
        # 条件 A: 检测到静音停顿
        pause_duration = current_time - self.last_speech_timestamp.get(meeting_id, current_time)
        if self.speech_buffer[meeting_id] and not has_speech and pause_duration > self.settings.VAD_PAUSE_DURATION_S:
            should_process = True
            reason = f"Silence pause ({pause_duration:.2f}s)"
            
        # 条件 B: 说话时间过长，强制转录一次 (防止长时间不触发)
        utterance_duration = current_time - self.buffer_start_time.get(meeting_id, current_time)
        MAX_DURATION = 15.0 # 硬编码 15 秒强制触发，也可放入 settings
        if self.speech_buffer[meeting_id] and utterance_duration > MAX_DURATION:
            should_process = True
            reason = f"Max duration reach ({utterance_duration:.2f}s)"

        if should_process:
            logger.info(f"Triggering transcription for {meeting_id}. Reason: {reason}")
            complete_utterance = np.concatenate(self.speech_buffer[meeting_id])
            
            # 清空缓冲区和重置时间戳
            self.speech_buffer[meeting_id] = []
            self.buffer_start_time[meeting_id] = current_time # 重置起始时间
            
            try:
                return await self._process_utterance(complete_utterance, sample_rate, stt_processor)
            except Exception as e:
                logger.error(f"Error processing utterance for meeting {meeting_id}: {e}", exc_info=True)
        
        return []

    async def _process_utterance(self, audio_data: np.ndarray, sample_rate: int, stt_processor: SpeechToTextProcessor) -> List[Dict[str, Any]]:
        """
        处理一个完整的语音片段（一句话）。
        - 进行说话人分离
        - 对每个语音片段进行STT
        - 识别或注册说话人
        """
        logger.info(f"Processing a complete utterance of {len(audio_data) / sample_rate:.2f}s.")
        
        if not self.embedding_model:
            logger.error("Embedding model not loaded, cannot process utterance.")
            return []

        waveform_tensor = torch.from_numpy(audio_data).float().to(self.device).unsqueeze(0)
        diarization = self.diarization_pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})

        results = []
        for segment, _, speaker_label in diarization.itertracks(yield_label=True):
            if segment.duration < self.settings.PYANNOTE_MIN_SPEECH_DURATION_S:
                continue # 跳过太短的片段

            segment_audio = audio_data[int(segment.start * sample_rate):int(segment.end * sample_rate)]
            
            # 1. 识别说话人
            identified_user = await self.identify_speaker(segment_audio, sample_rate)
            final_user_info = identified_user

            # 2. 如果是未知用户，则自动注册
            if not identified_user.get("is_known"):
                logger.info(f"Unknown speaker ({speaker_label}). Attempting to register new voiceprint.")
                new_user_id = str(uuid.uuid4())
                new_username = f"用户_{new_user_id[:6]}"
                new_role = UserRole.GUEST.value
                try:
                    reg_result = await self.register_voice(
                        audio_data=segment_audio,
                        sample_rate=sample_rate,
                        user_id=new_user_id,
                        username=new_username,
                        role=new_role
                    )
                    if reg_result.get("status") == "registered":
                        final_user_info = {"user_id": new_user_id, "username": new_username, "role": new_role, "is_known": True, "confidence": 100}
                        logger.info(f"New user '{new_username}' registered successfully.")
                    else:
                        logger.warning("Automatic registration failed.")
                except Exception as e:
                    logger.error(f"Error during automatic registration: {e}", exc_info=True)

            # 3. 进行语音转文字
            transcription_result = await stt_processor.transcribe_audio(segment_audio, sample_rate)
            transcribed_text = transcription_result.get("text", "")

            if transcribed_text:
                logger.info(f"User '{final_user_info.get('username')}' said: '{transcribed_text}'")
                results.append({
                    "audio": segment_audio,
                    "sample_rate": sample_rate,
                    "start_time": time.time() - (len(audio_data) / sample_rate) + segment.start,
                    "end_time": time.time() - (len(audio_data) / sample_rate) + segment.end,
                    "user_id": final_user_info.get("user_id"),
                    "username": final_user_info.get("username"),
                    "role": final_user_info.get("role"),
                    "is_new_user": not identified_user.get("is_known"),
                    "text": transcribed_text,
                    "confidence": transcription_result.get("confidence", 0.0),
                })
        
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

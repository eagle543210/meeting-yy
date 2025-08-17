# M:\meeting\core\speech_to_text\stt_processor.py

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import asyncio
import torch
from faster_whisper import WhisperModel
from abc import ABC, abstractmethod
import wave 
import time 

# 配置日志
logger = logging.getLogger(__name__)

# 定义抽象基类，以保持模块的可扩展性
class BaseSpeechToTextProcessor(ABC):
    @abstractmethod
    def __init__(self, settings_obj: Any):
        """初始化处理器，需要一个设置对象。"""
        pass

    @abstractmethod
    async def load_model(self):
        """异步加载模型。"""
        pass

    @abstractmethod
    async def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """
        转录音频数据。
        返回包含转录结果的字典，或者在没有有效转录时返回None。
        """
        pass

class SpeechToTextProcessor(BaseSpeechToTextProcessor):
    """
    负责语音转文本（STT）处理。
    使用 faster_whisper 库进行高效的音频转录。
    该实现已整合了更健壮的音频预处理、VAD 过滤和置信度计算逻辑。
    """
    def __init__(self, settings_obj: Any):
        """
        初始化处理器并设置模型参数。
        
        Args:
            settings_obj: 包含所有配置设置的对象。
        """
        self.settings = settings_obj
        self.model: Optional[WhisperModel] = None
        self.is_ready: bool = False
        self.is_loaded: bool = False
        self.logger = logging.getLogger(self.__class__.__name__)
        # 使用 getattr 安全地获取配置，如果不存在则使用默认值
        self.device = getattr(self.settings, 'WHISPER_MODEL_DEVICE', "cpu")
        self.compute_type = getattr(self.settings, 'STT_COMPUTE_TYPE', "float32")

        self.logger.info(f"SpeechToTextProcessor 初始化中。目标设备: {self.device}, 计算类型: {self.compute_type}.")

    async def load_model(self):
        """
        异步加载 Faster Whisper 模型。
        """
        if self.is_loaded:
            self.logger.info("模型已加载，无需重复加载。")
            self.is_ready = True
            return

        self.logger.info("开始加载 Whisper 模型...")
        
        # 优先使用 CUDA，如果可用
        device = "cuda" if torch.cuda.is_available() and getattr(self.settings, 'USE_CUDA', False) else "cpu"
        self.logger.info(f"模型设备：{device}")

        try:
            # 检查模型路径是否存在，确保从本地加载
            if not getattr(self.settings, 'WHISPER_MODEL_PATH', None):
                self.logger.error("未配置 WHISPER_MODEL_PATH，无法加载模型。")
                self.is_ready = False
                return
            
            # --- 显式将 pathlib.Path 对象转换为字符串，以确保与库兼容 ---
            model_path_str = str(self.settings.WHISPER_MODEL_PATH)
            
            self.logger.info(f"准备从本地路径加载 Whisper STT 模型: '{model_path_str}'...")
            
            # 使用 asyncio.to_thread 在线程中执行同步加载操作
            self.model = await asyncio.to_thread(
                WhisperModel,
                model_size_or_path=model_path_str,
                device=device,
                compute_type="float32",
                local_files_only=True # 仅从本地加载
            )
            self.is_loaded = True
            self.is_ready = True
            self.logger.info("Whisper 模型加载成功。")

        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}", exc_info=True)
            self.is_loaded = False
            self.is_ready = False

    def is_model_loaded(self) -> bool:
        """
        检查 STT 模型是否已成功加载并准备就绪。
        此方法用于兼容外部调用，例如 app.py。
        """
        return self.is_ready and self.model is not None

    async def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """
        执行音频转录。
        该方法包含了稳健的预处理、静音/过短音频检查和优化的置信度计算。
        
        Args:
            audio_data: 音频数据，numpy数组格式。
            sample_rate: 音频采样率。
        
        Returns:
            包含转录结果的字典，或者在没有有效转录时返回None。
        """
        if not self.is_ready or self.model is None:
            self.logger.error("STT 模型未加载或服务未准备好，无法执行转录。")
            return None # 修改：模型未准备好时，返回 None

        try:
            # --- 1. 音频预处理和静音/过短音频检测 ---
            audio_data = audio_data.squeeze()
            
            # faster-whisper 模型需要 float32 类型，且值在 [-1, 1] 范围
            if audio_data.dtype == np.int16:
                # 将 int16 数据转换为 float32 并归一化到 [-1, 1] 范围
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32:
                # 如果是其他类型，也需要转换为 float32，但不需要归一化
                audio_data = audio_data.astype(np.float32)

            # 检查无效数据
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                self.logger.warning("检测到无效音频数据 (NaN/inf)。")
                return None # 修改：无效音频数据时，返回 None
            
            # 音频过短检查
            if len(audio_data) < int(0.5 * sample_rate):
                self.logger.debug("音频过短，跳过转录。")
                return None # 修改：音频过短时，返回 None
            
            # --- 2. 设置转录参数 ---
            suppress_tokens = self._get_suppress_tokens()
            
            # 从 settings 中获取转录参数
            beam_size = getattr(self.settings, 'STT_BEAM_SIZE', 5)
            temperature = getattr(self.settings, 'STT_TEMPERATURE', 0.0)
            best_of = getattr(self.settings, 'STT_BEST_OF', 5)
            patience = getattr(self.settings, 'STT_PATIENCE', 1.0)
            
            self.logger.info(f"转录参数: beam_size={beam_size}, "
                             f"temperature={temperature}, "
                             f"suppress_tokens={suppress_tokens}")
            
            # --- 3. 调用转录，并启用内置 VAD 过滤 ---
            segments_generator, info = await asyncio.to_thread(
                self.model.transcribe,
                audio_data,
                beam_size=beam_size,
                language="zh",
                suppress_tokens=suppress_tokens,
                temperature=temperature,
                best_of=best_of,
                patience=patience,
                vad_filter=True,  # 核心：使用 faster-whisper 的内置 VAD
                vad_parameters=dict(
                    # 这是 VAD 的灵敏度阈值，已进一步降低默认值
                    threshold=getattr(self.settings, 'VAD_SPEECH_THRESHOLD', 0.05),
                    # 这是最小语音持续时间，也已降低默认值
                    min_speech_duration_ms=int(getattr(self.settings, 'MIN_SPEECH_DURATION_OFF', 0.8) * 1000),
                    max_speech_duration_s=20,
                    min_silence_duration_ms=500
                )
            )
            
            segments_list = list(segments_generator)
            
            if not segments_list:
                self.logger.warning("Faster Whisper 转录未检测到任何语音片段。")
                return None
            
            # --- 4. 处理转录结果 ---
            full_text = " ".join(seg.text for seg in segments_list)
            confidence = self._calculate_confidence(segments_list)
            
            # *** 修改：默认关闭置信度过滤 ***
            enable_confidence_filter = getattr(self.settings, 'STT_ENABLE_CONFIDENCE_FILTER', False)
            stt_confidence_threshold = getattr(self.settings, 'STT_CONFIDENCE_THRESHOLD', 0.3)
            
            if enable_confidence_filter and confidence < stt_confidence_threshold:
                self.logger.warning(f"转录置信度过低 (置信度: {confidence:.2f}), 已忽略。")
                return None

            segments_details = [{
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "probability": getattr(s, 'avg_logprob', 0.0),
                "no_speech_prob": getattr(s, 'no_speech_prob', 0.5)
            } for s in segments_list]
            
            self.logger.info(f"成功转录。文本: '{full_text}', 置信度: {confidence:.2f}, 语言: {info.language}")
            
            return {
                "text": full_text,
                "confidence": confidence,
                "language": info.language,
                "segments": segments_details
            }
            
        except Exception as e:
            self.logger.error(f"转录失败: {str(e)}", exc_info=True)
            return None

    def _get_suppress_tokens(self) -> Optional[List[int]]:
        """
        处理 suppress_tokens 参数，将其转换为模型能识别的 token 列表。
        """
        if not getattr(self.settings, 'STT_SUPPRESS_TOKENS', None):
            return None
            
        try:
            if hasattr(self.model, 'tokenizer'):
                return list(self.model.tokenizer.encode("<|notimestamps|>"))
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'tokenizer'):
                return list(self.model.model.tokenizer.encode("<|notimestamps|>"))
            else:
                return [-1]
        except Exception as e:
            self.logger.error(f"设置 suppress_tokens 失败: {e}")
            return None

    def _calculate_confidence(self, segments: List[Any]) -> float:
        """
        计算一个更可靠的转录置信度，综合考虑无语音概率和文本长度。
        """
        confidences = []
        for seg in segments:
            base_conf = 1.0 - getattr(seg, 'no_speech_prob', 0.5)
            
            if hasattr(seg, 'avg_logprob'):
                base_conf = (base_conf + seg.avg_logprob) / 2
            
            text_len = len(getattr(seg, 'text', ''))
            if text_len > 0:
                length_factor = min(1.0, text_len / 10.0)
                base_conf *= length_factor
                
            confidences.append(base_conf)
        
        return min(max(np.mean(confidences) if confidences else 0.0, 0.0), 1.0)
    
    async def close(self):
        """
        关闭 STT 模型，释放资源。
        """
        self.logger.info("SpeechToTextProcessor 正在关闭...")
        self.model = None
        self.is_ready = False
        self.is_loaded = False
        self.logger.info("SpeechToTextProcessor 已关闭。")

# M:\meeting\core\speech_to_text\stt_processor.py

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import asyncio
import torch
from faster_whisper import WhisperModel
from abc import ABC, abstractmethod
from pathlib import Path
import traceback

# 配置日志
logger = logging.getLogger(__name__)


# 抽象基类
class BaseSpeechToTextProcessor(ABC):
    @abstractmethod
    async def load_model(self):
        pass

    @abstractmethod
    async def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        pass


class SpeechToTextProcessor(BaseSpeechToTextProcessor):
    def __init__(self, settings_obj, logger=None, window_size=30.0, window_overlap=5.0):
        """
        初始化 STT 处理器
        """
        # 使用标准的 logging 模块，而非 print
        self.logger = logger or logging.getLogger(__name__)
        self.settings = settings_obj
        self.model = None
        self._model_loaded = False
        self.window_size = window_size
        self.window_overlap = window_overlap
        
    async def load_model(self):
        """
        根据配置文件动态加载 CPU/GPU 模型。
        """
        self._model_loaded = False
        self.model = None
    
        try:
            model_path = getattr(self.settings, "WHISPER_MODEL_PATH", None)
    
            if not model_path:
                raise RuntimeError("❌ 配置文件缺少 WHISPER_MODEL_PATH")
    
            model_path = Path(model_path)

            # --- 动态决定设备和计算类型 ---
            device_to_use = 'cpu'
            compute_type_to_use = getattr(self.settings, 'STT_COMPUTE_TYPE', 'int8')
    
            # 检查配置文件是否允许使用 CUDA 并且 CUDA 可用
            if getattr(self.settings, 'USE_CUDA', False) and torch.cuda.is_available():
                device_index = getattr(self.settings, 'WHISPER_MODEL_DEVICE_INDEX', 0)
                device_to_use = f"cuda:{device_index}"
                
                # 如果配置文件没有显式指定 compute_type，则默认为 float16
                if compute_type_to_use == 'int8':
                    compute_type_to_use = 'float16'
    
            self.logger.info(f"✅ 尝试加载模型到设备: {device_to_use}, 计算类型: {compute_type_to_use}")
    
            self.model = await asyncio.to_thread(
                WhisperModel,
                str(model_path),
                device=device_to_use,
                compute_type=compute_type_to_use,
                local_files_only=True
            )
            
            self.logger.info(f"✅ 模型加载成功: {model_path} (设备: {device_to_use})")
            self._model_loaded = True
            return self._model_loaded
    
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}\n{traceback.format_exc()}")
            self._model_loaded = False
            raise RuntimeError("❌ 无法加载任何可用的 STT 模型。") from e
    
    def is_model_loaded(self):
        return self._model_loaded
        
    def transcribe(self, audio_data, beam_size=5, temperature=0.0, suppress_tokens=None):
        """原始接口，兼容老调用"""
        return self.transcribe_audio(audio_data, beam_size=beam_size, temperature=temperature, suppress_tokens=suppress_tokens)

    async def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, beam_size: int = 5, temperature: float = 0.0, suppress_tokens: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        安全转录音频，兼容 VAD filter 移除所有音频。
        此方法为异步方法，并在任何情况下都返回一个字典。
        :param audio_data: numpy.ndarray, float32, shape=(samples,)
        :param sample_rate: 音频采样率 (默认为16000)
        """
        
        # 从 settings 中读取所有相关参数
        stt_beam_size = getattr(self.settings, 'STT_BEAM_SIZE', beam_size)
        stt_temperature = getattr(self.settings, 'STT_TEMPERATURE', temperature)
        stt_best_of = getattr(self.settings, 'STT_BEST_OF', 5)
        stt_patience = getattr(self.settings, 'STT_PATIENCE', 1.0)
        stt_language = getattr(self.settings, 'STT_LANGUAGE', 'zh')
        
        self.logger.info(f"转录参数: beam_size={stt_beam_size}, temperature={stt_temperature}, best_of={stt_best_of}, patience={stt_patience}, language='{stt_language}'")

        if not self._model_loaded or self.model is None:
            self.logger.error("❌ 模型未加载，无法转录音频")
            return {"text": "", "confidence": 0.0, "segments": []}

        try:
            if not isinstance(audio_data, np.ndarray):
                self.logger.error(f"❌ audio_data 必须是 numpy.ndarray, got {type(audio_data)}")
                return {"text": "", "confidence": 0.0, "segments": []}

            if audio_data.size == 0 or np.all(audio_data == 0):
                self.logger.warning("⚠️ 音频为空或全零，返回空转录")
                return {"text": "", "confidence": 0.0, "segments": []}

            # 确保音频数据为 float32
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_data = audio_data.astype(np.float32)

            self.logger.debug(f"音频 shape: {audio_data.shape}, dtype: {audio_data.dtype}")

            # 滑动窗口转录
            window_samples = int(self.window_size * sample_rate)
            overlap_samples = int(self.window_overlap * sample_rate)
            start = 0
            final_text_list = []
            final_segments_list = []
            total_confidence = 0
            segment_count = 0

            while start < len(audio_data):
                end = min(start + window_samples, len(audio_data))
                segment = audio_data[start:end]

                if np.all(segment == 0):
                    self.logger.debug(f"窗口 {start}-{end} 全零音频，跳过")
                    start += window_samples - overlap_samples
                    continue
                
                self.logger.debug(f"调用 faster-whisper 进行转录 (beam_size={stt_beam_size}, best_of={stt_best_of}, temperature={stt_temperature}, language='{stt_language}')")
                
                try:
                    segments_generator, info = await asyncio.to_thread(
                        self.model.transcribe,
                        segment,
                        beam_size=stt_beam_size,
                        temperature=stt_temperature,
                        suppress_tokens=self._get_suppress_tokens(),
                        language=stt_language,
                        best_of=stt_best_of,
                        patience=stt_patience,
                        vad_filter=True,  # 启用内置的 VAD 过滤
                    )
                    segments = list(segments_generator)
                    self.logger.debug(f"Faster Whisper 转录完成。检测到的语言信息: {info.language}, 概率: {info.language_probability:.4f}")
                except Exception as e:
                    self.logger.error(f"窗口 {start}-{end} 转录失败: {e}\n{traceback.format_exc()}")
                    segments = []

                if segments:
                    for s in segments:
                        if not s.text.strip():
                            self.logger.debug(f"Segment 文本为空，可能是静音或噪音，跳过。")
                            continue
                        
                        final_text_list.append(s.text)
                        final_segments_list.append(s)
                        total_confidence += s.avg_logprob
                        segment_count += 1
                else:
                    self.logger.debug(f"窗口 {start}-{end} VAD 移除所有音频，返回空")

                start += window_samples - overlap_samples

            final_text = " ".join(final_text_list).strip()
            average_confidence = total_confidence / segment_count if segment_count > 0 else 0.0

            # --- 幻觉过滤逻辑 ---
            # 幻觉置信度越接近0，正常置信度越低（负值更大）
            CONFIDENCE_UPPER_BOUND = -0.1
            if average_confidence > CONFIDENCE_UPPER_BOUND:
                self.logger.warning(f"⚠️ 由于最终平均置信度({average_confidence:.4f})接近于0，判定为幻觉并过滤掉。")
                # 返回一个空文本和置信度，但不是 None
                return {"text": "", "confidence": float(average_confidence), "segments": []}
            # --- 过滤逻辑结束 ---

            self.logger.info(f"最终拼接的转录文本: '{final_text}' (平均置信度: {average_confidence:.4f})")
            return {"text": final_text, "confidence": float(average_confidence), "segments": final_segments_list}

        except Exception as e:
            self.logger.error(f"❌ 转录失败: {e}\n{traceback.format_exc()}")
            return {"text": "", "confidence": 0.0, "segments": []}

    def is_model_loaded(self):
        return self._model_loaded

    def _get_suppress_tokens(self) -> Optional[List[int]]:
        if not getattr(self.settings, 'STT_SUPPRESS_TOKENS', None):
            return None
        try:
            if hasattr(self.model, 'tokenizer'):
                return list(self.model.tokenizer.encode("<|notimestamps|>"))
            else:
                return [-1]
        except Exception as e:
            self.logger.error(f"设置 suppress_tokens 失败: {e}")
            return None

    async def close(self):
        self.logger.info("SpeechToTextProcessor 正在关闭...")
        self.model = None
        self._model_loaded = False
        self.logger.info("SpeechToTextProcessor 已关闭。")
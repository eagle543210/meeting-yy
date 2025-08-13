# services\stt_service.py
import logging
import numpy as np
import os
from typing import Dict, Any, Optional

# 尝试导入 faster-whisper 的 WhisperModel
try:
    from faster_whisper import WhisperModel
except ImportError:
    logging.warning("faster-whisper 未安装，语音转文本功能将不可用。请运行 pip install faster-whisper")
    WhisperModel = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class STTService:
    """
    负责加载和管理 faster-whisper STT 模型，并执行音频转录。
    """
    def __init__(self, settings_obj: Any):
        self.settings = settings_obj
        self.model: Optional[WhisperModel] = None
        self.is_ready: bool = False
        self.device = "cpu"
        self.compute_type = "int8"
        
        if self.settings.USE_CUDA:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.compute_type = "float16"
                    logger.info("检测到 CUDA 可用，STT 模型将尝试加载到 GPU。")
                else:
                    logger.warning("settings.USE_CUDA 为 True，但 CUDA 不可用。STT 模型将加载到 CPU。")
            except ImportError:
                logger.warning("未安装 PyTorch，无法检查 CUDA。STT 模型将加载到 CPU。")
        
        logger.info(f"STTService 初始化中，模型待加载。目标设备: {self.device}。")
        
        if WhisperModel is None:
            error_msg = "faster-whisper 未安装，无法加载模型。请检查您的环境。"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            # 使用 settings.WHISPER_MODEL_PATH 作为模型路径，并设置 local_files_only=True
            # faster-whisper 的 WhisperModel 第一个参数可以是模型名称，也可以是本地模型目录的路径
            # 如果是本地路径，它会直接从该路径加载 CTranslate2 转换后的模型文件。
            # 您的模型文件在 \models\tiny，所以直接将这个路径作为模型参数。
            model_path_to_load = self.settings.WHISPER_MODEL_PATH 
            
            logger.info(f"正在加载 Whisper STT 模型: '{model_path_to_load}' (设备: {self.device}, 计算类型: {self.compute_type}, 强制本地文件加载: True)...")
            
            self.model = WhisperModel(
                model_path_to_load, # 直接传入本地模型目录的完整路径
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True # 强制只从本地文件加载，不尝试连接 Hugging Face Hub
            )
            self.is_ready = True
            logger.info(f"Whisper STT '{model_path_to_load}' 模型已成功加载到 {self.device}。")

        except Exception as e:
            error_msg = f"加载 Whisper STT 模型失败: {e}。请检查模型文件是否完整、本地路径是否正确，以及相关库版本兼容性。"
            logger.critical(error_msg, exc_info=True)
            self.model = None
            self.is_ready = False
            raise RuntimeError(error_msg)


    def is_model_loaded(self) -> bool:
        """
        检查 STT 模型是否已成功加载并准备就绪。
        """
        return self.is_ready and self.model is not None

    async def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        执行音频数据到文本的转录。
        参数:
            audio_data: NumPy 数组，音频数据。
            sample_rate: 音频的采样率。
        返回:
            包含转录文本和置信度的字典。
        """
        if not self.is_ready or self.model is None:
            logger.error("STT 模型未加载或服务未准备好，无法执行转录。")
            return {"text": "STT 服务未启用或模型未加载。", "confidence": 0.0, "segments": []}

        logger.info(f"开始转录音频。输入音频数据形状: {audio_data.shape}, 数据类型: {audio_data.dtype}, 采样率: {sample_rate} Hz。")
        if audio_data.size == 0:
            logger.warning("接收到空的音频数据数组，无法转录。")
            return {"text": "", "confidence": 0.0, "segments": []}
        
        try:
            if audio_data.dtype != np.float32:
                logger.debug(f"音频数据类型不是 float32，正在转换为 float32 (原始类型: {audio_data.dtype})。")
                audio_data = audio_data.astype(np.float32)
            else:
                logger.debug("音频数据类型已经是 float32，无需转换。")
            
            logger.info(f"调用 faster-whisper 进行转录 (beam_size=1, vad_filter=True, language='zh')...")
            logger.debug(f"音频数据（前10个样本）: {audio_data[:10]}")
            logger.debug(f"音频数据（后10个样本）: {audio_data[-10:]}")
            logger.debug(f"音频数据值范围: min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}")


            segments_generator, info = self.model.transcribe(
                audio_data,
                beam_size=1,
                vad_filter=True,
                language="zh"
            )
            logger.info(f"Faster Whisper 转录完成。检测到的语言信息: {info.language}, 概率: {info.language_probability:.4f}")

            full_transcript_segments = []
            segment_confidences = []
            
            segments_list = list(segments_generator) 

            for i, segment in enumerate(segments_list):
                logger.info(f"Faster Whisper Segment {i+1}: [{segment.start:.2f}s --> {segment.end:.2f}s] \"{segment.text}\" (Prob: {segment.avg_logprob:.2f}, No Speech Prob: {segment.no_speech_prob:.2f})")
                full_transcript_segments.append(segment.text)
                segment_confidences.append(segment.avg_logprob)

            text = " ".join(full_transcript_segments).strip()
            
            confidence = np.mean(segment_confidences) if segment_confidences else 0.0

            logger.info(f"Faster Whisper 最终拼接的转录文本: '{text}' (平均置信度: {confidence:.4f})")

            segments_details = [{"start": s.start, "end": s.end, "text": s.text, "probability": s.avg_logprob} for s in segments_list]
            
            logger.info(f"STT 转录结果返回：文本长度 {len(text)}，置信度 {confidence:.4f}。")
            
            return {"text": text, "confidence": float(confidence), "segments": segments_details}
        except Exception as e:
            logger.error(f"Whisper 音频转录失败: {e}", exc_info=True)
            return {"text": "转录过程中发生错误。", "confidence": 0.0, "segments": []}

    async def close(self):
        """
        关闭 STT 模型，释放资源。
        """
        logger.info("STTService 正在关闭...")
        if hasattr(self.model, 'release_resources') and callable(self.model.release_resources):
            self.model.release_resources()
            logger.info("Whisper 模型资源已释放。")
        self.model = None
        self.is_ready = False
        logger.info("STTService 已关闭。")

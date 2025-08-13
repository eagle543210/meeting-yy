# services/realtime_audio_processor.py
import torch
import numpy as np
import time
from collections import deque
import os
from typing import Any, Dict, Optional, List
from datetime import datetime
import logging
from config.settings import settings as app_settings 

logger = logging.getLogger(__name__)

class RealtimeAudioProcessor:
    """
    RealtimeAudioProcessor 负责接收实时的音频数据块，执行语音活动检测 (VAD)，
    缓冲检测到的语音片段，并将其传递给 BackendCoordinator 进行进一步的 STT、
    说话人识别和会议内容处理。
    """
    def __init__(self, config: app_settings, backend_coordinator):
        logger.info("初始化 RealtimeAudioProcessor...")
        self.settings = config
        self.backend_coordinator = backend_coordinator # BackendCoordinator 实例
        
        # 模型实例（由 BackendCoordinator 提供，此处不直接加载）
        self.vad_model = None 
        self.stt_processor = None 
        self.device = None # 设备信息，从 BackendCoordinator 获取

        # VAD 配置
        self.vad_sr = self.settings.VOICE_SAMPLE_RATE # VAD 和其他语音处理都使用统一的采样率
        self.vad_frame_size = self.settings.VAD_FRAME_SIZE # VAD 处理的帧大小
        self.vad_speech_threshold = self.settings.VAD_SPEECH_THRESHOLD # VAD 语音阈值
        self.vad_speech_timeout = self.settings.VAD_SPEECH_TIMEOUT # 静音超时，用于结束一个语音段
        self.vad_last_speech_end = time.time() # 上次检测到语音的结束时间

        # 音频缓冲区
        self.incoming_audio_queue = deque() # 原始传入音频的队列 (float32)
        self.speech_segment_buffer = np.array([], dtype=np.float32) # 累积语音片段用于发送给后端处理

        # STT/处理状态
        self.last_process_time = time.time() # 上次语音片段处理的时间
        # 控制将语音片段发送给后端处理的最小间隔（秒），避免过于频繁
        self.process_interval_seconds = self.settings.AUDIO_INTERVAL_SECONDS 
        
        logger.info("RealtimeAudioProcessor 初始化完成。")

    async def initialize(self):
        """
        异步初始化方法，用于从 BackendCoordinator 获取依赖组件。
        """
        logger.info("RealtimeAudioProcessor 正在异步初始化...")
        
        # 从 BackendCoordinator 获取 VAD 模型和 STT 处理器
        self.vad_model = getattr(self.backend_coordinator, 'vad_model', None)
        if not self.vad_model:
            logger.critical("VAD 模型未就绪。请确保它已在 BackendCoordinator 中加载。")
            raise RuntimeError("VAD 模型未就绪，RealtimeAudioProcessor 无法启动。")

        self.stt_processor = getattr(self.backend_coordinator, 'stt_processor', None)
        if not self.stt_processor or not self.stt_processor.is_model_loaded():
            logger.critical("STT 处理器未就绪或模型未加载。请确保它已在 BackendCoordinator 中初始化。")
            raise RuntimeError("STT 处理器未就绪，RealtimeAudioProcessor 无法启动。")
        
        self.device = getattr(self.backend_coordinator, 'device',
                              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        logger.info(f"RealtimeAudioProcessor 初始化完成。运行在设备: {self.device}")

    async def process_audio_chunk(self, audio_bytes: bytes, client_id: str, meeting_id: str):
        """
        处理传入的音频数据块。
        Args:
            audio_bytes (bytes): 客户端发送的原始音频字节数据 (通常是 INT16 PCM)。
            client_id (str): 发送音频的客户端ID。
            meeting_id (str): 当前会议的ID。
        """
        logger.debug(f"收到 {len(audio_bytes)} 字节音频数据 (客户端: {client_id})。")
        
        # 将 INT16 字节数据转换为 float32 numpy 数组，并归一化到 [-1.0, 1.0]
        try:
            audio_chunk_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"转换音频字节到 NumPy 数组失败 (客户端: {client_id}): {e}", exc_info=True)
            return

        if audio_chunk_np.size == 0:
            logger.warning(f"收到空音频块 (客户端: {client_id})。")
            return

        # 将当前音频块添加到队列中
        self.incoming_audio_queue.extend(audio_chunk_np)
        
        try:
            # 持续处理队列中的音频数据，直到不足一个 VAD 帧
            while len(self.incoming_audio_queue) >= self.vad_frame_size:
                # 每次从队列中取出一个 VAD 帧
                frame = np.array([self.incoming_audio_queue.popleft() for _ in range(self.vad_frame_size)])
                
                # VAD (语音活动检测)
                is_speech = False
                if self.vad_model:
                    # 将 NumPy 帧转换为 Torch Tensor 并移动到正确设备
                    audio_tensor = torch.from_numpy(frame).float().to(self.device)
                    with torch.no_grad():
                        speech_prob = self.vad_model(audio_tensor, self.vad_sr).item()
                        is_speech = speech_prob > self.vad_speech_threshold
                    # logger.debug(f"VAD 帧处理。概率: {speech_prob:.2f}, 是否语音: {is_speech}")
                else:
                    logger.warning("VAD 模型未就绪，默认将所有音频视为语音。")
                    is_speech = True # 如果 VAD 未加载，则将所有音频视为语音

                current_time = time.time()

                if is_speech:
                    self.vad_last_speech_end = current_time # 更新上次检测到语音的时间
                    self.speech_segment_buffer = np.concatenate((self.speech_segment_buffer, frame)) # 累积语音片段
                    
                    # 周期性地处理语音片段 (用于不间断的实时转录)
                    # 如果缓冲区累积到足够长度，并且距离上次处理时间已超过一定间隔
                    if (len(self.speech_segment_buffer) >= int(self.vad_sr * self.process_interval_seconds) and
                        (current_time - self.last_process_time > self.process_interval_seconds * 0.8)):
                        
                        logger.debug(f"周期性语音片段处理触发 (客户端: {client_id})。")
                        # 异步调用 BackendCoordinator 来处理这个语音片段
                        await self._process_current_speech_segment(client_id, meeting_id, is_final=False)
                        self.last_process_time = current_time # 重置时间
                        # self.speech_segment_buffer 在 _process_current_speech_segment 中被清空
                else:
                    # 检测到静音，检查是否超过静音超时阈值
                    if (len(self.speech_segment_buffer) > 0 and 
                        (current_time - self.vad_last_speech_end > self.vad_speech_timeout)):
                        
                        logger.info(f"静音超时触发 (客户端: {client_id})。正在最终处理语音片段。")
                        # 异步调用 BackendCoordinator 来处理最终的语音片段
                        await self._process_current_speech_segment(client_id, meeting_id, is_final=True)
                        self.last_process_time = current_time # 重置时间
                        # self.speech_segment_buffer 在 _process_current_speech_segment 中被清空
        
        except Exception as e:
            logger.error(f"处理音频块时发生错误 (客户端: {client_id}): {e}", exc_info=True)
            # 向客户端发送错误消息
            if hasattr(self.backend_coordinator, 'connection_manager') and self.backend_coordinator.connection_manager:
                await self.backend_coordinator.connection_manager.send_personal_message(
                    {
                        "type": "error",
                        "message": f"实时音频处理错误: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    },
                    client_id
                )

    async def _process_current_speech_segment(self, client_id: str, meeting_id: str, is_final: bool):
        """
        处理累积的语音片段，将其传递给 BackendCoordinator 进行 STT 和声纹识别。
        Args:
            client_id (str): 发言的客户端ID。
            meeting_id (str): 会议ID。
            is_final (bool): 指示这是否是语音活动的最终片段。
        """
        if len(self.speech_segment_buffer) == 0:
            logger.debug(f"语音片段缓冲区为空 (客户端: {client_id})。跳过处理。")
            return

        # 获取语音片段数据并清空缓冲区
        audio_segment = np.array(self.speech_segment_buffer, dtype=np.float32)
        self.speech_segment_buffer = np.array([], dtype=np.float32) # 立即清空缓冲区

        logger.debug(f"正在将语音片段 ({len(audio_segment)} 采样) 传递给 BackendCoordinator (客户端: {client_id}, 最终: {is_final})。")
        
        try:
            # 调用 BackendCoordinator 的方法，它将进一步处理这个语音片段 (STT, 声纹识别, 摘要等)
            # BackendCoordinator 将负责向前端广播结果
            await self.backend_coordinator.process_realtime_audio_segment(
                audio_data=audio_segment,
                sample_rate=self.vad_sr, # 使用 VAD 的采样率
                client_id=client_id,
                meeting_id=meeting_id,
                is_final=is_final
            )
        except Exception as e:
            logger.error(f"将语音片段传递给 BackendCoordinator 失败 (客户端: {client_id}): {e}", exc_info=True)
            # 可以在这里选择重新填充缓冲区或丢弃当前片段
            # 为避免无限循环，这里选择丢弃，并在 BackendCoordinator 中处理错误广播

    async def close(self):
        """
        清理 RealtimeAudioProcessor 内部的资源。
        由于模型由 BackendCoordinator 管理，这里主要清理缓冲区。
        """
        logger.info("RealtimeAudioProcessor 正在关闭...")
        self.incoming_audio_queue.clear()
        self.speech_segment_buffer = np.array([], dtype=np.float32)
        logger.info("RealtimeAudioProcessor 资源已清理。")

# live_mic_test.py
import sounddevice as sd
import numpy as np
import time
import os
import torch
import soundfile as sf
import logging

from config.settings import settings 

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pyannote 和 Faster Whisper 模型的导入
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

# --- 配置部分 (现在所有配置都从 settings 对象中获取) ---

# Hugging Face 访问 Token
# 从 SecretStr 获取实际值，如果不存在则为 None
hf_token = settings.HF_TOKEN.get_secret_value() if settings.HF_TOKEN else None

# 音频捕获设置
SAMPLE_RATE = settings.VOICE_SAMPLE_RATE

DURATION = 5 # 每次麦克风捕获的音频时长 (秒)
logger.warning(f"注意: AUDIO_CAPTURE_DURATION_SECONDS 未在 settings.py 中明确定义用于此脚本，当前使用默认值 {DURATION} 秒。")

CHANNELS = 1 # 麦克风通常捕获单声道

# Faster Whisper 模型设置
WHISPER_MODEL_PATH = str(settings.WHISPER_MODEL_PATH) # 使用 settings 中定义的模型路径
# settings.py 中有 STT_COMPUTE_TYPE 和 STT_BEAM_SIZE
COMPUTE_TYPE = settings.STT_COMPUTE_TYPE 
BEAM_SIZE = settings.STT_BEAM_SIZE
STT_LANGUAGE = settings.STT_LANGUAGE

# 设备选择
DEVICE = "cuda" if settings.USE_CUDA and torch.cuda.is_available() else "cpu"

logger.info(f"配置加载完成：")
logger.info(f"  HF_HUB_OFFLINE: {settings.HF_HUB_OFFLINE}")
logger.info(f"  Pyannote Diarization Model: {settings.PYANNOTE_DIARIZATION_MODEL}")
logger.info(f"  Whisper Model Local Path: {WHISPER_MODEL_PATH}")
logger.info(f"  Audio Sample Rate: {SAMPLE_RATE} Hz")
logger.info(f"  Audio Capture Duration: {DURATION} seconds")
logger.info(f"  Device: {DEVICE.upper()}")
logger.info(f"  Compute Type: {COMPUTE_TYPE}")
logger.info(f"  Whisper Beam Size: {BEAM_SIZE}")
logger.info(f"  Whisper Language: {STT_LANGUAGE}")

# --- 模型加载 ---
logger.info(f"正在加载 Pyannote 说话人分离模型 ({settings.PYANNOTE_DIARIZATION_MODEL})...")
try:
    # Pyannote 的 from_pretrained 方法会根据 settings 中加载的 HF_HUB_OFFLINE 环境变量来决定是否从本地缓存加载
    pyannote_pipeline = Pipeline.from_pretrained(settings.PYANNOTE_DIARIZATION_MODEL, use_auth_token=hf_token)
    pyannote_pipeline.to(torch.device(DEVICE))
    logger.info(f"Pyannote 模型已成功从本地缓存加载到 {DEVICE}。")
except Exception as e:
    logger.error(f"错误: 无法加载 Pyannote 模型！")
    logger.error(f"请确认模型 '{settings.PYANNOTE_DIARIZATION_MODEL}' 已被下载并存在于标准 Hugging Face 缓存目录中 (通常是 'C:\\Users\\Administrator\\.cache\\huggingface\\hub').")
    logger.error(f"或者检查您的 .env 文件或环境变量，确保 HF_HUB_OFFLINE 设置正确。")
    logger.error(f"详细错误: {e}")
    exit()

logger.info(f"正在加载 Faster Whisper 模型 (本地路径: {WHISPER_MODEL_PATH})...")
try:
    # 直接传入本地模型路径
    whisper_model = WhisperModel(WHISPER_MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info(f"Faster Whisper 模型已成功从本地路径加载到 {DEVICE}。")
except Exception as e:
    logger.error(f"错误: 无法从本地路径加载 Faster Whisper 模型！")
    logger.error(f"请确认 '{WHISPER_MODEL_PATH}' 路径下包含完整的 Faster Whisper 模型文件 (如 model.bin, vocabulary.txt 等)。")
    logger.error(f"详细错误: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 音频捕获函数 ---
def capture_audio(duration=DURATION, samplerate=SAMPLE_RATE, channels=CHANNELS):
    logger.info(f"\n开始录音 {duration} 秒...请说话")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='float32')
    sd.wait()
    logger.info("录音结束。")
    if channels == 1:
        audio_data = audio_data.flatten()
    return audio_data

# --- 主测试循环 ---
logger.info("\n--- 实时麦克风语音处理测试 ---")
logger.info(f"每次捕获 {DURATION} 秒音频。")
logger.info("按 Ctrl+C 停止。")

try:
    while True:
        live_audio = capture_audio()
        live_audio = live_audio.astype(np.float32)

        temp_audio_file = f"temp_live_audio_{int(time.time())}.wav"
        try:
            sf.write(temp_audio_file, live_audio, SAMPLE_RATE)
            logger.info(f"已保存当前捕获音频到: {temp_audio_file}")
        except Exception as e:
            logger.error(f"错误: 无法保存临时音频文件: {e}")
            continue

        logger.info("正在进行 Pyannote 说话人分离...")
        try:
            diarization = pyannote_pipeline(temp_audio_file) 
            logger.info("\n--- Pyannote 说话人分离结果 ---")
            
            speech_segments = []
           
            for segment, label in diarization.itertracks(): # 注意这里是 segment, label
                # segment 是 pyannote.core.Segment 对象
                # label 假设直接是说话人标签字符串 (如 'speaker_0', 'speaker_1')
                
                segment_duration = segment.end - segment.start
                logger.info(f"  说话人: {label}, 时间: {segment.start:.2f}s - {segment.end:.2f}s (时长: {segment_duration:.2f}s)")
                
                start_sample = int(segment.start * SAMPLE_RATE)
                end_sample = int(segment.end * SAMPLE_RATE)
                
                if end_sample <= len(live_audio):
                    speech_segment_audio = live_audio[start_sample:end_sample]
                    if speech_segment_audio.size > 0: 
                        speech_segments.append({
                            "speaker": label,
                            "start": segment.start,
                            "end": segment.end,
                            "audio": speech_segment_audio
                        })
                else:
                    logger.warning(f"  警告: 片段 {segment.start:.2f}-{segment.end:.2f} 越界，无法提取音频。")

            if not speech_segments:
                logger.info("  Pyannote 未检测到任何语音活动。")

            logger.info("\n--- Faster Whisper 转录结果 ---")
            if speech_segments:
                for seg_info in speech_segments:
                    speaker = seg_info["speaker"]
                    start_t = seg_info["start"]
                    end_t = seg_info["end"]
                    segment_audio_data = seg_info["audio"]

                    if len(segment_audio_data) == 0:
                        logger.warning(f"  说话人 {speaker} (时间: {start_t:.2f}s - {end_t:.2f}s): 音频片段为空，跳过转录。")
                        continue

                    logger.info(f"  正在转录说话人 {speaker} (时间: {start_t:.2f}s - {end_t:.2f}s) 的音频...")
                    
                    segments_whisper, info_whisper = whisper_model.transcribe(
                        segment_audio_data, 
                        beam_size=BEAM_SIZE,
                        vad_filter=True,
                        language=STT_LANGUAGE
                    )

                    transcribed_text = ""
                    total_confidence = 0
                    segment_count = 0
                    for s in segments_whisper:
                        transcribed_text += s.text.strip()
                        total_confidence += s.avg_logprob
                        segment_count += 1
                    
                    avg_confidence = total_confidence / segment_count if segment_count > 0 else 0

                    if transcribed_text:
                        logger.info(f"  [说话人: {speaker}, {start_t:.2f}s-{end_t:.2f}s] 转录: '{transcribed_text}' (置信度: {avg_confidence:.2f})")
                    else:
                        logger.warning(f"  [说话人: {speaker}, {start_t:.2f}s-{end_t:.2f}s] 未转录到文本 (置信度: {avg_confidence:.2f})。")
            else:
                logger.info("未检测到语音片段进行转录。")
        
        except Exception as e:
            logger.error(f"处理错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            
        logger.info("\n------------------------------------")
        logger.info(f"等待 {DURATION} 秒后再次录音，或者按 Ctrl+C 停止。")
        time.sleep(1)

except KeyboardInterrupt:
    logger.info("\n用户中断，测试结束。")
except Exception as e:
    logger.error(f"\n发生错误: {e}")
    import traceback
    traceback.print_exc()
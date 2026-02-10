import pyaudio
import numpy as np
import time
import sys
import logging
from faster_whisper import WhisperModel

# 设置日志级别以减少冗余输出
logging.basicConfig(level=logging.INFO)

# --- 1. 配置音频和模型参数 ---
# 定义音频流参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Faster Whisper模型所需的采样率
CHUNK = 1024  # 音频帧大小，每次读取的音频数据量

# ----------------------------
# 配置模型路径和设备
# IMPORTANT: 现在脚本会直接从这个本地路径加载模型
# 请确保你的模型文件存在于此路径下
# ----------------------------
MODEL_PATH = "M:\\meeting\\models\\tiny"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # 在大多数CPU上都能运行，且速度不错

print(f"正在加载 Whisper 模型 from {MODEL_PATH}...")
try:
    model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("模型加载成功。")
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit()

def record_and_transcribe():
    """
    从麦克风录音并进行实时转录。
    """
    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 打开音频流
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("麦克风已准备好，请开始说话...")
        print("按 Ctrl+C 停止。")
    except Exception as e:
        print(f"无法打开麦克风：{e}")
        print("请检查你的麦克风是否已连接，并且 `pyaudio` 库是否正确安装。")
        sys.exit()

    try:
        audio_buffer = []
        while True:
            # 读取音频数据
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_buffer.append(data)

            # 缓冲达到一定大小时进行处理 (例如，1秒的音频)
            if len(audio_buffer) >= RATE / CHUNK:
                # 将字节数据转换为 NumPy 数组
                audio_data = np.frombuffer(b''.join(audio_buffer), dtype=np.int16)

                # --- 核心改动：将数据类型从 int16 转换为 float32 ---
                # faster-whisper 模型需要浮点数输入，这里将 int16 归一化到 [-1, 1] 范围
                audio_data = audio_data.astype(np.float32) / 32768.0

                # 清空缓冲区以开始新的批次
                audio_buffer = []

                # --- 转录部分 ---
                segments, info = model.transcribe(audio_data, language="zh", vad_filter=True, beam_size=5)

                for segment in segments:
                    # 打印转录结果
                    print(f"[转录] {segment.text}")

    except KeyboardInterrupt:
        print("\n\n转录已停止。")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 停止并关闭音频流和 PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    record_and_transcribe()

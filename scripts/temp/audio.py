import os
import torch
from dotenv import load_dotenv


os.environ["HF_HUB_OFFLINE"] = "1"


from pyannote.audio import Pipeline

load_dotenv()

HF_AUTH_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN") 

device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"检测到的设备: {device_str.upper()}")
device = torch.device(device_str)

model_id = "pyannote/speaker-diarization-3.1"
print(f"尝试以离线模式加载模型: {model_id}。将完全从本地缓存加载。")

try:
    
    pipeline = Pipeline.from_pretrained(
        model_id, 
      
    ) 
    
    pipeline.to(device)

    print(f"\n🎉 恭喜！模型已成功以离线模式加载。")
    print("\nPipeline 对象类型:", type(pipeline))
    if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'hparams'):
        print("Pipeline 配置样例 (部分键):", pipeline.model.hparams.keys())
    else:
        print("无法访问 pipeline.model.hparams，但模型似乎已加载。")

  
    test_audio_file = "M:/8.wav" 
    
    if os.path.exists(test_audio_file):
        print(f"\n正在对音频文件 '{test_audio_file}' 进行说话人分离推理...")
        diarization = pipeline(test_audio_file)
        
        print("\n说话人分离结果：")
        for segment, track, label in diarization.itertracks(yield_label=True):
            print(f"  {segment.start:.1f}s - {segment.end:.1f}s: Speaker {label}")
        
        output_rttm_path = "output_diarization.rttm"
        with open(output_rttm_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
        print(f"\n说话人分离结果已保存到 '{output_rttm_path}'")

    else:
        print(f"\n警告：未找到测试音频文件 '{test_audio_file}'，跳过推理测试。")
        print("请准备一个 .wav 音频文件并更新 `test_audio_file` 变量以进行实际测试。")
        print("确保它是单声道，16kHz采样率。")

except Exception as e:
    print(f"\n❌ 错误：模型离线加载或推理失败！")
    print(f"错误信息: {e}")
    print("\n请确认以下事项：")
    print("1. **模型是否已在有网络时成功缓存过一次。** 这是最关键的前提。")
    print("2. `HF_HUB_OFFLINE` 环境变量是否在 `pyannote.audio` 导入前设置。")
    print("3. 如果您要使用 GPU，请确保 CUDA 和 PyTorch GPU 版本兼容。")
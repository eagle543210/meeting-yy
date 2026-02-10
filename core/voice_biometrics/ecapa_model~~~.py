# M:\meeting\core\voice_biometrics\ecapa_model.py

import torch
import torchaudio 
import torchaudio.transforms as T
import numpy as np
import logging
import os
from typing import Optional

from speechbrain.pretrained import EncoderClassifier
from config.settings import settings

logger = logging.getLogger(__name__)

class ECAPASpeakerEmbedding:
    def __init__(self, device: str = "cpu"):
        # 获取模型本地目录。
        # 用户必须在 settings.py 中正确配置 ECAPA_MODEL_DIR
        # default_model_dir = "M:/meeting/models/ecapa_tdnn" 
        self.MODEL_LOCAL_DIR = os.path.abspath(getattr(settings, 'ECAPA_MODEL_DIR'))
        
        # 模型的预期采样率，确保与模型训练时一致
        self.sample_rate = getattr(settings, 'VOICE_SAMPLE_RATE', 16000)

        # 确保本地模型目录存在
        os.makedirs(self.MODEL_LOCAL_DIR, exist_ok=True)
        logger.info(f"ECAPA 模型将从目录: '{self.MODEL_LOCAL_DIR}' 加载。")
        if not os.path.isdir(self.MODEL_LOCAL_DIR):
            logger.error(f"无法创建或访问 ECAPA 模型目录: {self.MODEL_LOCAL_DIR}")
            raise FileNotFoundError(f"ECAPA 模型目录 '{self.MODEL_LOCAL_DIR}' 无法创建或访问。请检查路径和权限。")

        self.device = torch.device("cuda" if torch.cuda.is_available() else device)
        logger.info(f"ECAPA 模型将运行在设备: {self.device}")

        self.classifier = None
        self.embedding_dim = getattr(settings, 'VOICE_EMBEDDING_DIM', 192) # 默认 ECAPA-TDNN 嵌入维度是 192

        try:
            logger.info(f"从本地目录 '{self.MODEL_LOCAL_DIR}' 加载 ECAPA-TDNN 模型")
            
            # 使用 SpeechBrain 的预训练模型加载方式,
            # source 是 Hugging Face Hub ID，即使本地加载也需要，用于识别模型类型
            # savedir 是模型文件的本地缓存路径
            # local_files_only=True 强制只使用本地文件，不会尝试从网络下载
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-tdnn", 
                savedir=self.MODEL_LOCAL_DIR,
                run_opts={"device": str(self.device)},
                # local_files_only=True # 强制仅使用本地文件，禁止网络下载
            )
            
            self.classifier.eval() # 将模型设置为评估模式，禁用 dropout 等

            # (可选) 验证模型实际输出的 embedding_dim
            # 这一段逻辑可以帮助你在初始化时检测配置是否与模型实际输出匹配
            # dummy_audio = torch.randn(1, self.sample_rate).to(self.device)
            # dummy_embedding = self.classifier.encode_batch(dummy_audio, normalize=True) 
            # actual_embedding_dim = dummy_embedding.shape[-1]
            # if actual_embedding_dim != self.embedding_dim:
            #     logger.warning(
            #         f"⚠️ settings.py 中的 VOICE_EMBEDDING_DIM ({self.embedding_dim}) "
            #         f"与模型实际输出维度 ({actual_embedding_dim}) 不匹配。 "
            #         f"建议更新 VOICE_EMBEDDING_DIM 为 {actual_embedding_dim}。"
            #     )
            #     self.embedding_dim = actual_embedding_dim

            logger.info(f"✅ ECAPA-TDNN 模型从 '{self.MODEL_LOCAL_DIR}' 加载成功，嵌入维度: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"❌ ECAPA-TDNN 模型加载失败: {e}", exc_info=True)
            self.classifier = None
            error_message = (
                f"ECAPA-TDNN 模型加载失败 (尝试从本地目录 '{self.MODEL_LOCAL_DIR}' 加载): {str(e)}\n\n"
                f"请严格按照以下步骤操作：\n"
                f"1. **确认配置**: 确保 `config/settings.py` 文件中的 `ECAPA_MODEL_DIR` 设置为正确的模型文件存放目录。\n"
                f"   当前配置的目录是: '{self.MODEL_LOCAL_DIR}'\n"
                f"2. **下载完整模型文件**: 从 Hugging Face Hub (speechbrain/spkrec-ecapa-tdnn) 下载所有必要文件。\n"
                f"   通常包括: 'hyperparams.yaml', 'embedding_model.ckpt', 'custom.py' (如果存在), 'label_encoder.txt' 等。\n"
                f"3. **放置文件**: 将下载的所有文件直接放置在上述 '{self.MODEL_LOCAL_DIR}' 目录中。\n"
                f"4. **检查文件和权限**: 确认文件未损坏，并且程序对该目录及其中的文件有足够的读取权限。\n\n"
                f"如果错误是由于文件缺失 (例如 'hyperparams.yaml not found')，请务必检查第2步和第3步。"
            )
            raise RuntimeError(error_message)

    def extract_features_from_buffer(self, audio_data: np.ndarray, input_sample_rate: int) -> Optional[np.ndarray]:
        # ... (方法开头部分保持不变，直到 raw_embedding_np = raw_embeddings.squeeze().cpu().numpy()) ...

        if self.classifier is None:
            logger.error("❌ ECAPA-TDNN 模型未加载。无法提取声纹。")
            return None

        if audio_data.size == 0:
            logger.warning("音频缓冲区为空，无法提取特征。")
            return None

        try:
            # 确保音频数据为 float32 类型
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # 将 numpy 数组转换为 PyTorch Tensor
            wav_tensor = torch.from_numpy(audio_data)

            # 统一音频张量的维度为 (batch, time)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0) # (time,) -> (1, time)
            elif wav_tensor.ndim == 2:
                # 假设是 (channels, time) 或 (time, channels)
                # 统一为 (1, time) 单声道
                if wav_tensor.shape[0] > 1: # 如果是多声道，取平均值
                    wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)
                else: # 如果是 (1, time) 格式，保持不变
                    pass
            else:
                logger.error(f"音频缓冲区维度不正确 ({wav_tensor.ndim}D)，预期为1D或2D。")
                return None

            # 检查音频数据范围并进行归一化（如果需要）
            if torch.max(torch.abs(wav_tensor)) > 1.0:
                logger.debug("音频数据值超出 [-1, 1] 范围，正在归一化。")
                wav_tensor = wav_tensor / torch.max(torch.abs(wav_tensor))

            if input_sample_rate != self.sample_rate:
                logger.debug(f"输入采样率 ({input_sample_rate}Hz) 与模型预期 ({self.sample_rate}Hz) 不匹配，正在重采样。")
                resampler = T.Resample(orig_freq=input_sample_rate, new_freq=self.sample_rate).to(self.device)
                wav_tensor = resampler(wav_tensor.to(self.device))
            else:
                wav_tensor = wav_tensor.to(self.device)

            with torch.no_grad():
                raw_embeddings = self.classifier.encode_batch(wav_tensor)

            raw_embedding_np = raw_embeddings.squeeze().cpu().numpy()

            # --- 核心修改：将 norm 的计算和 L2 归一化放在这里 ---
            norm = np.linalg.norm(raw_embedding_np)
            if norm == 0:
                normalized_embedding = raw_embedding_np
                logger.warning(
                    f"ECAPA 嵌入向量的 L2 范数为零，未进行归一化。请检查音频质量或模型输出。"
                )
            else:
                normalized_embedding = raw_embedding_np / norm


            # --- 打印原始嵌入日志 (现在在归一化之前) ---
            logger.info(
                f"ECAPA 原始嵌入向量（numpy）形状: {raw_embedding_np.shape}, "
                f"min: {raw_embedding_np.min():.4f}, max: {raw_embedding_np.max():.4f}, "
                f"mean: {raw_embedding_np.mean():.4f}, std: {raw_embedding_np.std():.4f}"
            )
            logger.info(f"ECAPA 原始嵌入向量（numpy）前5个值: {raw_embedding_np[:5]}")

            # --- 打印归一化后的日志 (现在在归一化之后，确保 normalized_embedding 已定义) ---
            logger.info(f"ECAPA L2 归一化后的向量范数: {np.linalg.norm(normalized_embedding):.4f}")
            logger.info(
                f"ECAPA L2 归一化嵌入向量（numpy）形状: {normalized_embedding.shape}, "
                f"min: {normalized_embedding.min():.4f}, max: {normalized_embedding.max():.4f}, "
                f"mean: {normalized_embedding.mean():.4f}, std: {normalized_embedding.std():.4f}"
            )
            logger.info(f"ECAPA L2 归一化嵌入向量（numpy）前5个值: {normalized_embedding[:5]}")


            if np.isnan(normalized_embedding).any() or np.isinf(normalized_embedding).any():
                logger.error("❌ 提取的声纹特征包含 NaN 或 Inf 值。")
                return None

            logger.debug(f"成功提取声纹嵌入，形状: {normalized_embedding.shape}")

            return normalized_embedding

        except Exception as e:
            logger.error(f"❌ 提取 ECAPA 声纹失败: {e}", exc_info=True)
            return None
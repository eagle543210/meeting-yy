# M:\meeting\services\embedding_service.py

import logging
from typing import List, Optional
import numpy as np
import torch
import asyncio
from pathlib import Path # 导入 Path
import os # 导入 os 用于环境变量

# 导入配置
from config.settings import settings 
logger = logging.getLogger(__name__)

# 尝试导入 transformers 库
try:
    from transformers import AutoTokenizer, AutoModel
    logger.info("transformers 库导入成功。")
except ImportError:
    logger.warning("无法导入 transformers 库。请确保已安装。BGE 嵌入模型功能将受限。")
    AutoTokenizer = None
    AutoModel = None

class BGEEmbeddingModel:
    """
    BGE (BAAI General Embedding) 嵌入模型服务，用于生成文本的向量嵌入。
    """
    def __init__(self, settings_obj: settings):
        logger.info("初始化 BGEEmbeddingModel...")
        self.settings = settings_obj
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        # 使用 settings.USE_CUDA 来决定设备
        self.device = "cuda" if self.settings.USE_CUDA and torch.cuda.is_available() else "cpu"
        logger.info(f"BGEEmbeddingModel 将使用设备: {self.device}")
        logger.info("BGEEmbeddingModel 初始化完成。")

    async def load_model(self):
        """
        异步加载 BGE 嵌入模型。
        """
        logger.info(f"BGEEmbeddingModel: 正在加载模型 '{self.settings.BGE_MODEL_NAME}' 到 {self.device}...")
        if AutoTokenizer is None or AutoModel is None:
            logger.error("transformers 库未导入，无法加载 BGE 模型。")
            return

        try:
            # 直接使用 settings.BGE_MODEL_PATH，它已经是 Path 对象
            local_model_path = self.settings.BGE_MODEL_PATH 
            
            # 检查本地路径是否存在且非空
            if local_model_path.exists() and any(local_model_path.iterdir()): # 检查目录非空
                logger.info(f"从本地路径加载 BGE 模型: {local_model_path}")
                self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, str(local_model_path))
                self.model = await asyncio.to_thread(AutoModel.from_pretrained, str(local_model_path))
            else:
                logger.info(f"本地路径 {local_model_path} 无模型文件或目录为空，尝试从 Hugging Face Hub 下载模型 '{self.settings.BGE_MODEL_NAME}'。")
                # 确保 HF_HUB_OFFLINE 环境变量已根据 settings.HF_HUB_OFFLINE 设置
                os.environ["HF_HUB_OFFLINE"] = self.settings.HF_HUB_OFFLINE

                self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, self.settings.BGE_MODEL_NAME)
                self.model = await asyncio.to_thread(AutoModel.from_pretrained, self.settings.BGE_MODEL_NAME)
                
                # 下载后保存到本地，以便下次离线加载
                logger.info(f"模型 '{self.settings.BGE_MODEL_NAME}' 下载完成，正在保存到本地: {local_model_path}")
                await asyncio.to_thread(self.tokenizer.save_pretrained, str(local_model_path))
                await asyncio.to_thread(self.model.save_pretrained, str(local_model_path))

            self.model.eval() # 设置模型为评估模式
            self.model.to(self.device) # 将模型移动到指定设备
            logger.info(f"✅ BGE 嵌入模型 '{self.settings.BGE_MODEL_NAME}' 已加载到 {self.device}。")
        except Exception as e:
            logger.critical(f"❌ BGE 嵌入模型加载失败: {e}", exc_info=True)
            self.tokenizer = None
            self.model = None
            raise RuntimeError(f"BGE 嵌入模型加载失败: {str(e)}")

    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载。
        """
        return self.tokenizer is not None and self.model is not None

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        生成给定文本的嵌入向量。
        Args:
            text (str): 输入文本。
        Returns:
            Optional[List[float]]: 文本的嵌入向量列表，如果模型未加载或生成失败则为 None。
        """
        if not self.is_model_loaded():
            logger.error("BGE 嵌入模型未加载。无法生成嵌入向量。")
            return None

        try:
            # 异步执行分词和模型推理
            inputs = await asyncio.to_thread(
                self.tokenizer,
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.settings.BART_MAX_INPUT_LENGTH # 使用 BART_MAX_INPUT_LENGTH 作为通用文本输入长度限制
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = await asyncio.to_thread(self.model, **inputs)
                # 获取 [CLS] token 的嵌入向量作为句子嵌入
                embeddings = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
            
            # 返回第一个（也是唯一一个）文本的嵌入向量
            return embeddings[0].tolist()

        except Exception as e:
            logger.error(f"生成文本嵌入失败: {e}", exc_info=True)
            return None

    async def close(self):
        """
        关闭 BGEEmbeddingModel 及其内部资源。
        """
        logger.info("BGEEmbeddingModel: 正在关闭...")
        self.tokenizer = None
        self.model = None
        # 释放 CUDA 内存 (如果使用 GPU)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("BGEEmbeddingModel 已关闭。")


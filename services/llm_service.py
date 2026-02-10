# M:\meeting\services\llm_service.py

import logging
from typing import Optional, List, Dict, Any, AsyncIterator, Iterator
import asyncio
import os
import torch
import json
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    logger.info("llama_cpp 库导入成功。")
except ImportError:
    logger.warning("无法导入 llama_cpp 库。请确保已安装。LLM 模型功能将受限。")
    Llama = None

class LLMModel:
    """
    大型语言模型 (LLM) 服务，用于文本生成、摘要、问答和实体提取。
    支持 GGUF 格式的模型。
    """
    def __init__(self, settings_obj: settings):
        logger.info("初始化 LLMModel...")
        self.settings = settings_obj
        self.model: Optional[Llama] = None
        self.device = "cuda"
        logger.info(f"LLMModel 将使用设备: {self.device}")
        logger.info("LLMModel 初始化完成。")

    async def load_model(self):
        """
        异步加载 LLM 模型。
        """
        logger.info(f"LLMModel: 正在加载模型 '{self.settings.LLM_MODEL_NAME}'...")
        if Llama is None:
            logger.error("llama_cpp 库未导入，无法加载 LLM 模型。")
            return

        try:
            model_file_path = self.settings.LLM_MODEL_PATH
            logger.info(f"LLMModel: 尝试从以下路径加载模型: {model_file_path}")

            if not model_file_path.exists():
                logger.critical(f"❌ LLM 模型文件不存在: {model_file_path}")
                raise FileNotFoundError(f"LLM 模型文件不存在于指定路径: {model_file_path}")

            logger.info(f"从本地路径加载 LLM 模型: {model_file_path}")
            
            n_gpu_layers = self.settings.LLAMA_N_GPU_LAYERS
            if not torch.cuda.is_available():
                n_gpu_layers = 0 # 如果没有 CUDA 或禁用 CUDA，则强制使用 CPU

            # 模型加载本身是同步的，使用 asyncio.to_thread 避免阻塞主事件循环
            self.model = await asyncio.to_thread(
                Llama,
                model_path=str(model_file_path),
                n_ctx=self.settings.LLAMA_N_CTX,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            logger.info(f"✅ LLM 模型 '{self.settings.LLM_MODEL_NAME}' 已加载。GPU 层数: {n_gpu_layers}")
        except Exception as e:
            logger.critical(f"❌ LLM 模型加载失败: {e}", exc_info=True)
            self.model = None
            raise RuntimeError(f"LLM 模型加载失败: {str(e)}")

    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载。
        """
        return self.model is not None

    def _sync_chat_generation(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> Iterator[str]:
        """
        同步的聊天生成函数，用于在另一个线程中运行。
        """
        try:
            stream_response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>"],
                stream=True
            )
            for chunk in stream_response:
                if "content" in chunk["choices"][0]["delta"]:
                    yield chunk["choices"][0]["delta"]["content"]
        except Exception as e:
            logger.error(f"LLM 聊天流生成失败: {e}", exc_info=True)
            raise

    async def generate_text(self, messages: List[Dict[str, str]], max_new_tokens: int = 200, temperature: float = 0.7) -> AsyncIterator[str]:
        """
        根据聊天消息列表生成文本响应，始终作为异步生成器。
        Args:
            messages (List[Dict[str, str]]): 包含 'system' 和 'user' 角色的消息列表。
            max_new_tokens (int): 生成的最大新 token 数量。
            temperature (float): 采样温度。
        Returns:
            AsyncIterator[str]: 生成的文本流。
        """
        if not self.is_model_loaded():
            logger.error("LLM 模型未加载。无法生成文本。")
            yield "对不起，LLM 模型尚未准备好。"
            return

        try:
            # 等待 to_thread 协程，获取同步迭代器，然后对其进行异步迭代
            stream_gen = await asyncio.to_thread(
                self._sync_chat_generation, 
                messages, 
                max_tokens=max_new_tokens, 
                temperature=temperature
            )
            for chunk in stream_gen:
                yield chunk
        except Exception as e:
            logger.error(f"LLM 文本流生成失败: {e}", exc_info=True)
            yield f"生成文本流时发生错误: {str(e)}"
    
    async def extract_action_items(self, text: str) -> List[str]:
        """
        从文本中提取行动项。此方法是非流式的。
        """
        # 构建一个专门用于提取行动项的聊天提示
        messages = [
            {"role": "system", "content": "你是一个会议纪要助手。请从提供的文本中提取明确的行动项。"},
            {"role": "user", "content": f"从以下会议文本中提取所有明确的行动项，并以列表形式返回。如果文本中没有明确的行动项，请返回空列表。只返回 JSON 列表。例如：['完成报告', '联系张三']。\n\n文本: {text}"}
        ]
        
        raw_response_chunks = []
        async for chunk in self.generate_text(
            messages,
            max_new_tokens=self.settings.LLM_ACTION_ITEMS_MAX_LENGTH,
            temperature=0.5,
        ):
            raw_response_chunks.append(chunk)
        raw_response = "".join(raw_response_chunks)

        try:
            action_items = json.loads(raw_response)
            if isinstance(action_items, list) and all(isinstance(item, str) for item in action_items):
                return action_items
            else:
                logger.warning(f"LLM 提取行动项返回非列表或非字符串列表格式: {raw_response}")
                return []
        except json.JSONDecodeError:
            logger.warning(f"LLM 提取行动项返回非 JSON 格式，尝试按行分割: {raw_response}")
            return [item.strip() for item in raw_response.split('\n') if item.strip()]
    
    async def extract_decisions(self, text: str) -> List[str]:
        """
        从文本中提取关键决策。此方法是非流式的。
        """
        messages = [
            {"role": "system", "content": "你是一个会议纪要助手。请从提供的文本中提取关键决策。"},
            {"role": "user", "content": f"从以下会议文本中提取所有关键决策，并以列表形式返回。如果文本中没有明确的决策，请返回空列表。只返回 JSON 列表。例如：['通过了新项目预算', '决定推迟发布日期']。\n\n文本: {text}"}
        ]
        
        raw_response_chunks = []
        async for chunk in self.generate_text(
            messages,
            max_new_tokens=self.settings.LLM_ACTION_ITEMS_MAX_LENGTH,
            temperature=0.5,
        ):
            raw_response_chunks.append(chunk)
        raw_response = "".join(raw_response_chunks)

        try:
            decisions = json.loads(raw_response)
            if isinstance(decisions, list) and all(isinstance(item, str) for item in decisions):
                return decisions
            else:
                logger.warning(f"LLM 提取决策返回非列表或非字符串列表格式: {raw_response}")
                return []
        except json.JSONDecodeError:
            logger.warning(f"LLM 提取决策返回非 JSON 格式，尝试按行分割: {raw_response}")
            return [item.strip() for item in raw_response.split('\n') if item.strip()]
    
    async def extract_topics(self, text: str) -> List[str]:
        """
        从文本中提取主要话题。此方法是非流式的。
        """
        messages = [
            {"role": "system", "content": "你是一个会议纪要助手。请从提供的文本中提取主要话题。"},
            {"role": "user", "content": f"从以下会议文本中提取 3-5 个主要话题或主题，并以列表形式返回。如果文本中没有明确的主题，请返回空列表。只返回 JSON 列表。例如：['项目进展', '市场分析', '预算讨论']。\n\n文本: {text}"}
        ]
        
        raw_response_chunks = []
        async for chunk in self.generate_text(
            messages,
            max_new_tokens=100,
            temperature=0.7,
        ):
            raw_response_chunks.append(chunk)
        raw_response = "".join(raw_response_chunks)

        try:
            topics = json.loads(raw_response)
            if isinstance(topics, list) and all(isinstance(item, str) for item in topics):
                return topics
            else:
                logger.warning(f"LLM 提取话题返回非列表或非字符串列表格式: {raw_response}")
                return []
        except json.JSONDecodeError:
            logger.warning(f"LLM 提取话题返回非 JSON 格式，尝试按逗号或分行分割: {raw_response}")
            return [item.strip() for item in raw_response.replace(',', '\n').split('\n') if item.strip()]

    async def answer_question(self, context: str, question: str) -> AsyncIterator[str]:
        """
        根据提供的上下文回答问题。此方法是流式的。
        """
        messages = [
            {"role": "system", "content": "你是一个问答助手。请根据提供的上下文回答问题。"},
            {"role": "user", "content": f"上下文: {context}\n\n问题: {question}"}
        ]
        
        async for chunk in self.generate_text(
            messages,
            max_new_tokens=self.settings.LLM_QA_MAX_LENGTH,
            temperature=0.7,
        ):
            yield chunk

    async def close(self):
        """
        关闭 LLMModel 及其内部资源。
        """
        logger.info("LLMModel: 正在关闭...")
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("LLMModel 已关闭。")

# services\llm_service.py

import logging
from typing import Optional, List, Dict, Any, AsyncIterator
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
        self.device = "cuda" if self.settings.USE_CUDA and torch.cuda.is_available() else "cpu"
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
            if not self.settings.USE_CUDA or not torch.cuda.is_available():
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

    async def generate_text(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, stream: bool = False) -> AsyncIterator[str]:
        """
        生成文本响应，始终作为异步生成器。
        Args:
            prompt (str): 输入提示。
            max_new_tokens (int): 生成的最大新 token 数量。
            temperature (float): 采样温度。
            stream (bool): 如果为 True，则以流式生成文本。
        Returns:
            AsyncIterator[str]: 生成的文本流。
        """
        if not self.is_model_loaded():
            logger.error("LLM 模型未加载。无法生成文本。")
            yield "对不起，LLM 模型尚未准备好。" # 直接 yield 错误消息
            return # 结束生成器

        if stream:
            try:
                # 在单独的线程中运行同步的 llama_cpp 流式生成，并逐块 yield
                for chunk in await asyncio.to_thread(
                    self.model.create_completion,
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=["<|im_end|>", "用户:", "User:"],
                    stream=True # 启用流式
                ):
                    if "choices" in chunk and len(chunk["choices"]) > 0 and "text" in chunk["choices"][0]:
                        yield chunk["choices"][0]["text"]
            except Exception as e:
                logger.error(f"LLM 文本流生成失败: {e}", exc_info=True)
                yield f"生成文本流时发生错误: {str(e)}"
            logger.debug("LLM 文本流生成完成。") # 仅在流结束时记录
        else:
            # 非流式调用，仍然使用 asyncio.to_thread，但将完整响应作为单个块 yield
            try:
                response = await asyncio.to_thread(
                    self.model.create_completion,
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=["<|im_end|>", "用户:", "User:"]
                )
                generated_text = response["choices"][0]["text"].strip()
                logger.debug(f"LLM 生成文本完成。")
                yield generated_text # <--- 修正: yield 完整的文本作为单个块
            except Exception as e:
                logger.error(f"LLM 文本生成失败: {e}", exc_info=True)
                yield f"生成文本时发生错误: {str(e)}"

    async def extract_action_items(self, text: str) -> List[str]:
        """
        从文本中提取行动项。此方法是非流式的。
        """
        if not self.is_model_loaded():
            logger.error("LLM 模型未加载。无法提取行动项。")
            return []
        
        prompt = f"请从以下会议文本中提取所有明确的行动项，并以列表形式返回。如果文本中没有明确的行动项，请返回空列表。例如：['完成报告', '联系张三']。\n\n文本: {text}\n\n行动项:"
        try:
            raw_response_chunks = []
            # 使用 async for 循环消费 generate_text 的输出
            async for chunk in self.generate_text(
                prompt, 
                max_new_tokens=self.settings.LLM_ACTION_ITEMS_MAX_LENGTH,
                temperature=0.5,
                stream=False # 非流式调用
            ):
                raw_response_chunks.append(chunk)
            raw_response = "".join(raw_response_chunks) # 将所有块连接起来 (对于非流式通常只有一个块)
            
            try:
                action_items = json.loads(raw_response)
                if isinstance(action_items, list) and all(isinstance(item, str) for item in action_items):
                    return action_items
                else:
                    logger.warning(f"LLM 提取行动项返回非列表或非字符串列表格式: {raw_response}")
                    return [item.strip() for item in raw_response.split('\n') if item.strip()]
            except json.JSONDecodeError:
                logger.warning(f"LLM 提取行动项返回非 JSON 格式，尝试按行分割: {raw_response}")
                return [item.strip() for item in raw_response.split('\n') if item.strip()]

        except Exception as e:
            logger.error(f"提取行动项失败: {e}", exc_info=True)
            return []

    async def extract_decisions(self, text: str) -> List[str]:
        """
        从文本中提取关键决策。此方法是非流式的。
        """
        if not self.is_model_loaded():
            logger.error("LLM 模型未加载。无法提取决策。")
            return []

        prompt = f"请从以下会议文本中提取所有关键决策，并以列表形式返回。如果文本中没有明确的决策，请返回空列表。例如：['通过了新项目预算', '决定推迟发布日期']。\n\n文本: {text}\n\n决策:"
        try:
            raw_response_chunks = []
            # 使用 async for 循环消费 generate_text 的输出
            async for chunk in self.generate_text(
                prompt, 
                max_new_tokens=self.settings.LLM_ACTION_ITEMS_MAX_LENGTH,
                temperature=0.5,
                stream=False # 非流式调用
            ):
                raw_response_chunks.append(chunk)
            raw_response = "".join(raw_response_chunks) # 将所有块连接起来
            
            try:
                decisions = json.loads(raw_response)
                if isinstance(decisions, list) and all(isinstance(item, str) for item in decisions):
                    return decisions
                else:
                    logger.warning(f"LLM 提取决策返回非列表或非字符串列表格式: {raw_response}")
                    return [item.strip() for item in raw_response.split('\n') if item.strip()]
            except json.JSONDecodeError:
                logger.warning(f"LLM 提取决策返回非 JSON 格式，尝试按行分割: {raw_response}")
                return [item.strip() for item in raw_response.split('\n') if item.strip()]

        except Exception as e:
            logger.error(f"提取决策失败: {e}", exc_info=True)
            return []

    async def extract_topics(self, text: str) -> List[str]:
        """
        从文本中提取主要话题。此方法是非流式的。
        """
        if not self.is_model_loaded():
            logger.error("LLM 模型未加载。无法提取话题。")
            return []

        prompt = f"请从以下会议文本中提取 3-5 个主要话题或主题，并以列表形式返回。例如：['项目进展', '市场分析', '预算讨论']。\n\n文本: {text}\n\n话题:"
        try:
            raw_response_chunks = []
            # 使用 async for 循环消费 generate_text 的输出
            async for chunk in self.generate_text(
                prompt, 
                max_new_tokens=100,
                temperature=0.7,
                stream=False # 非流式调用
            ):
                raw_response_chunks.append(chunk)
            raw_response = "".join(raw_response_chunks) # 将所有块连接起来
            
            try:
                topics = json.loads(raw_response)
                if isinstance(topics, list) and all(isinstance(item, str) for item in topics):
                    return topics
                else:
                    logger.warning(f"LLM 提取话题返回非列表或非字符串列表格式: {raw_response}")
                    return [item.strip() for item in raw_response.replace(',', '\n').split('\n') if item.strip()]
            except json.JSONDecodeError:
                logger.warning(f"LLM 提取话题返回非 JSON 格式，尝试按逗号或分行分割: {raw_response}")
                return [item.strip() for item in raw_response.replace(',', '\n').split('\n') if item.strip()]

        except Exception as e:
            logger.error(f"提取话题失败: {e}", exc_info=True)
            return []

    async def answer_question(self, context: str, question: str) -> AsyncIterator[str]:
        """
        根据提供的上下文回答问题，支持流式输出。
        Args:
            context (str): 用于回答问题的上下文文本。
            question (str): 用户提出的问题。
        Returns:
            AsyncIterator[str]: 问题的答案流。
        """
        if not self.is_model_loaded():
            logger.error("LLM 模型未加载。无法回答问题。")
            yield "对不起，LLM 模型尚未准备好，无法回答您的问题。" # 直接 yield 错误消息
            return # 结束生成器

        logger.info(f"LLMModel: 正在使用 LLM 回答问题 (流式)...")
        prompt = f"根据以下上下文回答问题。如果上下文没有足够的信息来回答问题，请说明。\n\n上下文: {context}\n\n问题: {question}\n\n答案:"
        try:
            # 调用 generate_text 并启用流式
            async for chunk in self.generate_text(
                prompt, 
                max_new_tokens=self.settings.LLM_QA_MAX_LENGTH,
                temperature=0.7,
                stream=True # 启用流式
            ):
                yield chunk
        except Exception as e:
            logger.error(f"LLM 问答流式生成失败: {e}", exc_info=True)
            yield f"回答问题时发生错误: {str(e)}"

    async def close(self):
        """
        关闭 LLMModel 及其内部资源。
        """
        logger.info("LLMModel: 正在关闭...")
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("LLMModel 已关闭。")


# services\summary_service.py

import logging
import os
import asyncio
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

# ******** 强制 Hugging Face Hub 进入离线模式 ********
# 这确保了模型会从本地缓存加载，而不是尝试从网络下载。
os.environ["HF_HUB_OFFLINE"] = "1"
# ******************************************************

# 从项目配置中导入设置
from config.settings import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummaryService:
    """
    负责文本摘要的服务类。
    """
    def __init__(self, settings_obj):
        """
        初始化 SummaryService。
        Args:
            settings_obj: 应用程序的设置对象，包含 SUMMARY_MODEL_PATH 等配置。
        """
        logger.info("正在初始化 SummaryService...")
        self.settings = settings_obj
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.pipeline_instance = None # 避免与 from transformers import pipeline 冲突
        self.device = "cpu" # 默认设备
        self.model_loaded = False # 初始模型加载状态

    async def load_model(self):
        """
        异步加载文本摘要模型。
        """
        if self.model_loaded:
            logger.info("摘要模型已加载，跳过重复加载。")
            return

        try:
            # 1. 检测可用设备 (CUDA 或 CPU)
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("检测到 CUDA，将使用 GPU 加载摘要模型。")
            else:
                self.device = "cpu"
                logger.info("未检测到 CUDA，将使用 CPU 加载摘要模型。")

            # 2. 尝试加载分词器
            logger.info(f"尝试从本地路径加载分词器: '{self.settings.SUMMARY_MODEL_PATH}'...")
            if not os.path.exists(self.settings.SUMMARY_MODEL_PATH):
                logger.critical(f"❌ 错误：摘要模型本地路径不存在: '{self.settings.SUMMARY_MODEL_PATH}'。请确保文件已正确放置。")
                raise FileNotFoundError(f"摘要模型本地路径不存在: {self.settings.SUMMARY_MODEL_PATH}")

            try:
                self.tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained,
                    self.settings.SUMMARY_MODEL_PATH, # 直接使用本地路径
                    local_files_only=True # 明确指定只从本地文件加载
                )
                logger.info("✅ 分词器加载成功。")
            except Exception as e:
                logger.critical(f"❌ 错误：分词器加载失败！请检查 '{self.settings.SUMMARY_MODEL_PATH}' 中的分词器文件是否完整且兼容。")
                logger.critical(f"错误信息 (分词器): {e}", exc_info=True)
                raise RuntimeError(f"无法加载分词器: {e}") from e

            # 3. 尝试加载模型
            logger.info(f"尝试从本地路径加载模型: '{self.settings.SUMMARY_MODEL_PATH}' (设备: {self.device})...")
            try:
                self.model = await asyncio.to_thread(
                    AutoModelForSeq2SeqLM.from_pretrained,
                    self.settings.SUMMARY_MODEL_PATH, # 直接使用本地路径
                    local_files_only=True # 明确指定只从本地文件加载
                )
                self.model.to(self.device) # 将模型移动到指定设备
                logger.info("✅ 模型加载成功。")
            except Exception as e:
                logger.critical(f"❌ 错误：模型加载失败！请检查 '{self.settings.SUMMARY_MODEL_PATH}' 中的模型文件是否完整且兼容。")
                logger.critical(f"错误信息 (模型): {e}", exc_info=True)
                raise RuntimeError(f"无法加载模型: {e}") from e

            # 4. 创建 Hugging Face pipeline
            logger.info("正在创建 Hugging Face pipeline...")
            try:
                self.pipeline_instance = await asyncio.to_thread(
                    pipeline,
                    "summarization", # 任务类型
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1 # 0 for GPU, -1 for CPU
                )
                logger.info("✅ Hugging Face pipeline 创建成功。")
            except Exception as e:
                logger.critical(f"❌ 错误：创建 Hugging Face pipeline 失败！")
                logger.critical(f"错误信息 (pipeline): {e}", exc_info=True)
                raise RuntimeError(f"无法创建 pipeline: {e}") from e

            self.model_loaded = True # This line is only reached if all above steps succeed
            logger.info(f"🎉 摘要模型 '{self.settings.SUMMARY_MODEL_HUB_NAME}' 已成功从本地路径加载到 {self.device.upper()}。")

        except RuntimeError as e: # 捕获内部抛出的 RuntimeError
            self.model_loaded = False # 确保在失败时设置为 False
            logger.critical(f"摘要模型加载流程中断: {e}")
            raise # 重新抛出，让 lifespan 捕获
        except Exception as e: # 捕获其他意外错误
            self.model_loaded = False # 确保在失败时设置为 False
            logger.critical(f"❌ 错误：摘要模型加载过程中发生意外错误！")
            logger.critical(f"错误信息: {e}", exc_info=True)
            raise RuntimeError(f"摘要模型加载失败: {e}") from e
        finally:
            # 无论成功或失败，都在这里记录最终状态
            logger.info(f"SummaryService.load_model() 方法结束。最终 model_loaded 状态: {self.model_loaded}")


    def is_model_loaded(self) -> bool:
        """
        检查摘要模型是否已加载。
        """
        return self.model_loaded

    async def generate_summary(self, text: str, max_length: int = 150, min_length: int = 30, num_beams: int = 4) -> str:
        """
        生成文本摘要。
        Args:
            text (str): 要摘要的文本。
            max_length (int): 生成摘要的最大长度。
            min_length (int): 生成摘要的最小长度。
            num_beams (int): Beam search 的束宽。
        Returns:
            str: 生成的摘要文本。
        """
        if not self.model_loaded or self.pipeline_instance is None:
            logger.error("摘要模型未加载，无法生成摘要。")
            return "错误：摘要服务未准备好。"

        try:
            # 异步执行摘要生成
            summary_list = await asyncio.to_thread(
                self.pipeline_instance, # 使用更名后的 pipeline_instance
                text,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                do_sample=False # 通常摘要是确定性的
            )
            summary_text = summary_list[0]['summary_text']
            logger.info(f"成功生成摘要 (长度: {len(summary_text)})。")
            return summary_text
        except Exception as e:
            logger.error(f"生成摘要失败: {e}", exc_info=True)
            return f"生成摘要失败: {e}"

    async def close(self):
        """
        关闭 SummaryService，释放模型资源。
        """
        logger.info("Closing SummaryService...")
        self.model = None
        self.tokenizer = None
        self.pipeline_instance = None
        self.model_loaded = False
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared for SummaryService.")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache for SummaryService: {e}")
        logger.info("SummaryService closed.")


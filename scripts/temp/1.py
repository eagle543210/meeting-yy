# verify_bart_load.py
import os
import logging
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ******** 强制 Hugging Face Hub 进入离线模式 ********
# 这确保了模型会从本地缓存加载，而不是尝试从网络下载。
os.environ["HF_HUB_OFFLINE"] = "1"
# ******************************************************

# 导入 settings 对象，以获取模型路径
try:
    from config.settings import settings
except ImportError:
    logging.critical("❌ 错误：无法导入 config.settings。请确保 config/settings.py 文件存在且可访问。")
    exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def verify_bart_model_load():
    """
    异步验证 BART 摘要模型在离线模式下是否能从本地路径加载。
    """
    model_path = settings.SUMMARY_MODEL_PATH
    model_name = settings.SUMMARY_MODEL_HUB_NAME # 用于日志显示

    logger.info(f"验证脚本启动。当前 HF_HUB_OFFLINE 环境变量: {os.environ.get('HF_HUB_OFFLINE')}")
    
    # 检测可用设备
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"检测到的设备: {device_str.upper()}")

    model_loaded_successfully = False

    # 1. 检查本地模型路径是否存在
    if not os.path.exists(model_path):
        logger.critical(f"❌ 错误：摘要模型本地路径不存在: '{model_path}'。请确保文件已正确放置。")
        logger.critical("请再次确认您手动下载的文件是否真的在上述路径下，并且文件夹结构正确。")
        return False

    logger.info(f"本地模型路径 '{model_path}' 存在。")

    # 2. 尝试加载分词器
    tokenizer = None
    try:
        logger.info(f"尝试从本地路径加载分词器: '{model_path}'...")
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained,
            model_path, # 直接使用本地路径
            local_files_only=True # 明确指定只从本地文件加载
        )
        logger.info("✅ 分词器加载成功。")
    except Exception as e:
        logger.critical(f"❌ 错误：分词器加载失败！请检查 '{model_path}' 中的分词器文件是否完整且兼容。")
        logger.critical(f"错误信息 (分词器): {e}", exc_info=True)
        return False

    # 3. 尝试加载模型
    model = None
    try:
        logger.info(f"尝试从本地路径加载模型: '{model_path}' (设备: {device_str})...")
        model = await asyncio.to_thread(
            AutoModelForSeq2SeqLM.from_pretrained,
            model_path, # 直接使用本地路径
            local_files_only=True # 明确指定只从本地文件加载
        )
        model.to(device) # 将模型移动到指定设备
        logger.info("✅ 模型加载成功。")
    except Exception as e:
        logger.critical(f"❌ 错误：模型加载失败！请检查 '{model_path}' 中的模型文件是否完整且兼容。")
        logger.critical(f"错误信息 (模型): {e}", exc_info=True)
        return False

    # 4. 创建 Hugging Face pipeline
    pipeline_instance = None
    try:
        logger.info("正在创建 Hugging Face pipeline...")
        pipeline_instance = await asyncio.to_thread(
            pipeline,
            "summarization", # 任务类型
            model=model,
            tokenizer=tokenizer,
            device=0 if device_str == "cuda" else -1 # 0 for GPU, -1 for CPU
        )
        logger.info("✅ Hugging Face pipeline 创建成功。")
    except Exception as e:
        logger.critical(f"❌ 错误：创建 Hugging Face pipeline 失败！")
        logger.critical(f"错误信息 (pipeline): {e}", exc_info=True)
        return False

    model_loaded_successfully = True
    logger.info(f"🎉 摘要模型 '{model_name}' 已成功从本地路径加载到 {device_str.upper()}。")

    # 5. 进行一个简单的摘要测试
    test_text = "这是一个测试文本，用于验证摘要模型是否正常工作。它应该能够生成一个简短的总结。"
    logger.info(f"\n正在对测试文本进行摘要: '{test_text}'")
    try:
        summary_list = await asyncio.to_thread(
            pipeline_instance,
            test_text,
            max_length=50,
            min_length=10,
            num_beams=4,
            do_sample=False
        )
        summary_text = summary_list[0]['summary_text']
        logger.info(f"✅ 摘要生成测试成功。生成的摘要: '{summary_text}'")
    except Exception as e:
        logger.critical(f"❌ 错误：摘要生成测试失败！即使模型加载成功，推理也可能出现问题。")
        logger.critical(f"错误信息 (摘要测试): {e}", exc_info=True)
        model_loaded_successfully = False # 标记为失败，因为推理不工作

    return model_loaded_successfully

if __name__ == "__main__":
    if asyncio.run(verify_bart_model_load()):
        logger.info("\n🎉 摘要模型已完全验证成功。")
    else:
        logger.critical("\n❌ 摘要模型验证失败。请检查上述日志以获取详细错误信息。")


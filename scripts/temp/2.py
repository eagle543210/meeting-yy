# M:\meeting\simulate_ingestion.py

import asyncio
import logging
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional

# 导入项目内部模块 (现在直接从 models 包导入)
from config.settings import settings
from services.mongodb_manager import MongoDBManager
from services.milvus_service import MilvusManager
from services.embedding_service import BGEEmbeddingModel
from models import TranscriptEntry, User, UserRole # 导入 User 和 UserRole
from pymilvus import connections # 导入 connections 用于统一管理连接

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # <--- 提高日志级别到 DEBUG
logger = logging.getLogger(__name__)

async def simulate_meeting_ingestion():
    """
    模拟会议内容的摄取过程，包括保存到 MongoDB 和插入到 Milvus。
    """
    logger.info("--- 开始模拟会议内容导入 ---")

    # 初始化服务
    mongodb_manager: Optional[MongoDBManager] = None
    meeting_milvus_manager: Optional[MilvusManager] = None
    bge_model: Optional[BGEEmbeddingModel] = None

    try:
        # 1. 初始化 MongoDBManager
        logger.info("正在初始化 MongoDBManager...")
        mongodb_manager = MongoDBManager(
            host=settings.MONGO_HOST,
            port=settings.MONGO_PORT,
            db_name=settings.MONGO_DB_NAME
        )
        await mongodb_manager.connect()
        if not mongodb_manager.is_connected:
            logger.critical("❌ MongoDBManager 连接失败，无法继续。")
            return
        logger.info("MongoDBManager 已连接。")

        # 2. 初始化会议文本嵌入 MilvusManager
        logger.info(f"正在初始化会议文本嵌入 MilvusManager (集合: {settings.MILVUS_MEETING_COLLECTION_NAME})...")
        meeting_milvus_manager = MilvusManager(
            config=settings,
            collection_name=settings.MILVUS_MEETING_COLLECTION_NAME,
            schema_fields=settings.MILVUS_MEETING_SCHEMA_FIELDS
        )
        # 强制覆盖集合，确保每次测试都是干净的开始
        # 如果您希望保留之前的数据，可以改为 overwrite_collection=False
        await meeting_milvus_manager.connect(overwrite_collection=True) # <--- 注意: 这里设置为 True 会清空现有数据
        if not meeting_milvus_manager.is_connected:
            logger.critical("❌ MilvusManager (会议文本嵌入) 连接失败，无法继续。")
            return
        logger.info("会议文本嵌入 MilvusManager 已初始化并连接。")

        # 3. 初始化 BGE 嵌入模型
        logger.info("正在初始化 BGE 嵌入模型...")
        bge_model = BGEEmbeddingModel(settings_obj=settings)
        await bge_model.load_model()
        if not bge_model.is_model_loaded():
            logger.critical("❌ BGE 嵌入模型未能成功加载，将无法生成文本嵌入，无法继续。")
            return
        logger.info("BGE 嵌入模型已加载。")

        # 模拟的用户数据
        simulated_users = {
            "zhangsan_id": {"username": "张三", "role": UserRole.HOST.value},
            "lisi_id": {"username": "李四", "role": UserRole.MEMBER.value},
            "wangwu_id": {"username": "王五", "role": UserRole.GUEST.value},
            "zhaoli_id": {"username": "赵丽", "role": UserRole.MEMBER.value},
        }

        logger.info("正在确保模拟用户在 MongoDB 中存在...")
        for user_id, user_info in simulated_users.items():
            user_obj = User(
                user_id=user_id, # user_id 会被映射到 _id
                username=user_info["username"],
                role=UserRole(user_info["role"].upper()),
                created_at=datetime.utcnow(), # 确保这些字段有值
                last_active=datetime.utcnow()
            )
            # 检查 add_or_update_user 的返回值
            success = await mongodb_manager.add_or_update_user(user_obj)
            if not success:
                logger.error(f"❌ 无法在 MongoDB 中注册/更新用户 '{user_info['username']}' (ID: {user_id})。")
            else:
                logger.info(f"用户 '{user_info['username']}' (ID: {user_id}) 已在 MongoDB 中注册/更新。")

        # 模拟会议内容
        meeting_id = str(uuid.uuid4()) # 为本次模拟生成一个唯一的会议ID
        logger.info(f"本次模拟的会议 ID: {meeting_id}")

        simulated_transcripts = [
            {"speaker_id": "zhangsan_id", "text": "各位同事，早上好。今天我们主要讨论一下关于新产品线上推广的方案。"},
            {"speaker_id": "lisi_id", "text": "张总，关于线上推广，我这边准备了一份针对短视频平台的详细计划，特别是抖音和快手。"},
            {"speaker_id": "wangwu_id", "text": "是的，短视频现在确实是流量高地，但我们需要考虑预算问题，以及如何精准触达目标用户。"},
            {"speaker_id": "zhangsan_id", "text": "预算方面，李四你那边的计划有详细的成本估算吗？我们需要确保投入产出比。"},
            {"speaker_id": "zhaoli_id", "text": "我建议我们可以先进行小范围的A/B测试，看看哪种策略效果最好，再决定大范围推广。同时，用户反馈模块的迭代也非常关键。"},
            {"speaker_id": "lisi_id", "text": "预算估算我已经做好了，预计初期投入在50万左右，主要用于KOL合作和广告投放。用户反馈模块的迭代，我同意赵丽的看法，这能帮助我们及时调整策略。"},
            {"speaker_id": "zhangsan_id", "text": "好的，那我们今天的结论就是：李四负责细化短视频平台的推广方案和预算报告；赵丽负责用户反馈模块的优先级排期和迭代计划。下周二我们再开会讨论具体进展。"},
            {"speaker_id": "wangwu_id", "text": "没问题，我会关注预算的合理性。"},
            {"speaker_id": "lisi_id", "text": "好的，张总，我会尽快完成。"},
            {"speaker_id": "zhaoli_id", "text": "我也会立即着手准备。"}
        ]

        current_time = datetime.utcnow()
        logger.info("正在处理模拟转录并插入数据库...")
        for i, entry_data in enumerate(simulated_transcripts):
            speaker_id = entry_data["speaker_id"]
            text = entry_data["text"]
            
            # 从 MongoDB 获取用户资料以获取正确的 username 和 role
            user_profile = await mongodb_manager.get_user(speaker_id)
            if not user_profile:
                logger.warning(f"模拟用户 '{speaker_id}' 未在 MongoDB 中找到，跳过此条转录。请检查用户注册是否成功。")
                continue

            # 创建 TranscriptEntry
            transcript_entry = TranscriptEntry(
                # id 会由 default_factory 自动生成 UUID 字符串
                meeting_id=meeting_id,
                client_id=speaker_id, # 模拟客户端ID与用户ID相同
                user_id=speaker_id,
                speaker_id=user_profile.username, # 使用 MongoDB 中的 username 作为 speaker_id
                role=user_profile.role.value,
                text=text,
                timestamp=current_time + timedelta(seconds=i*5) # 模拟时间递增
            )

            # 1. 保存转录到 MongoDB 并获取 _id
            mongo_doc_id = await mongodb_manager.save_transcript_entry(transcript_entry)
            if not mongo_doc_id:
                logger.error(f"❌ 无法将转录保存到 MongoDB: {text}。跳过 Milvus 插入。")
                continue
            logger.info(f"转录已保存到 MongoDB (MongoDB ID: {mongo_doc_id}, 发言人: {user_profile.username})")

            # 2. 生成文本嵌入并插入 Milvus
            text_embedding = await bge_model.get_embedding(text)
            if text_embedding:
                milvus_data_entry = {
                    "embedding": text_embedding,
                    "mongo_id": str(mongo_doc_id) # 确保 mongo_id 是字符串
                }
                try:
                    pks = await meeting_milvus_manager.insert_data([milvus_data_entry])
                    if pks:
                        logger.info(f"文本嵌入已插入 Milvus 集合 '{settings.MILVUS_MEETING_COLLECTION_NAME}' (Milvus ID: {pks[0]}, MongoDB ID: {mongo_doc_id})")
                    else:
                        logger.error(f"❌ 插入文本嵌入到 Milvus 失败，未返回主键 (MongoDB ID: {mongo_doc_id})。")
                except Exception as e:
                    logger.error(f"❌ 插入文本嵌入到 Milvus 失败 (MongoDB ID: {mongo_doc_id}): {e}", exc_info=True)
            else:
                logger.warning(f"❌ 无法生成文本嵌入，跳过 Milvus 插入 (MongoDB ID: {mongo_doc_id})。")

        logger.info("--- 模拟会议内容导入完成 ---")
        logger.info(f"请使用此会议 ID 进行 RAG 测试: {meeting_id}")

    except Exception as e:
        logger.critical(f"模拟导入过程中发生严重错误: {e}", exc_info=True)
    finally:
        # 确保 Milvus 集合被释放，而不是全局断开连接
        if meeting_milvus_manager and meeting_milvus_manager.collection:
            try:
                await asyncio.to_thread(meeting_milvus_manager.collection.release)
                logger.info(f"Milvus 集合 '{settings.MILVUS_MEETING_COLLECTION_NAME}' 已释放。")
            except Exception as e:
                logger.warning(f"释放 Milvus 集合 '{settings.MILVUS_MEETING_COLLECTION_NAME}' 失败: {e}")

        # 关闭所有服务
        if bge_model:
            await bge_model.close()
            logger.info("BGE 嵌入模型已关闭。")
        if meeting_milvus_manager:
            # 这里的 close 方法只关闭 manager 内部的 collection 引用，不关闭全局连接
            await meeting_milvus_manager.close()
            logger.info("会议文本嵌入 MilvusManager 已关闭。")
        if mongodb_manager:
            await mongodb_manager.close()
            logger.info("MongoDBManager 已关闭。")
        
        # 全局 Milvus 连接在 app.py 的 lifespan 中统一管理，这里不进行 connections.disconnect()

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(simulate_meeting_ingestion())


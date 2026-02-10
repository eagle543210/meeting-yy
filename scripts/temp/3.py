# M:\meeting\verify_data.py

import asyncio
import logging
from typing import Optional, List, Dict, Any

# 导入项目内部模块 (现在直接从 models 包导入)
from config.settings import settings
from services.mongodb_manager import MongoDBManager
from services.milvus_service import MilvusManager
from pymilvus import Collection # 直接导入 Collection 类以获取实体数量
from models import User # 导入 User 模型用于验证用户数据

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def verify_database_data():
    """
    验证 MongoDB 和 Milvus 集合中是否存在数据。
    """
    logger.info("--- 开始验证数据库数据 ---")

    # 初始化服务
    mongodb_manager: Optional[MongoDBManager] = None
    voice_milvus_manager: Optional[MilvusManager] = None
    meeting_milvus_manager: Optional[MilvusManager] = None

    try:
        # 1. 初始化 MongoDBManager
        logger.info("正在初始化 MongoDBManager...")
        mongodb_manager = MongoDBManager(
            host=settings.MONGO_HOST,
            port=settings.MONGO_PORT,
            db_name=settings.MONGO_DB_NAME
        )
        await mongodb_manager.connect()
        logger.info("MongoDBManager 已连接。")

        # 验证 MongoDB 中的集合
        logger.info(f"--- 验证 MongoDB 集合 '{settings.MONGO_TRANSCRIPT_COLLECTION_NAME}' (会议转录) ---")
        transcript_count = await mongodb_manager.db[settings.MONGO_TRANSCRIPT_COLLECTION_NAME].count_documents({})
        logger.info(f"MongoDB 集合 '{settings.MONGO_TRANSCRIPT_COLLECTION_NAME}' 中有 {transcript_count} 条文档。")
        if transcript_count > 0:
            logger.info("✅ MongoDB 会议转录数据存在。")
            sample_transcripts = await mongodb_manager.db[settings.MONGO_TRANSCRIPT_COLLECTION_NAME].find().limit(3).to_list()
            for i, doc in enumerate(sample_transcripts):
                logger.info(f"  示例转录 {i+1}: Speaker ID: {doc.get('speaker_id')}, Text: {doc.get('text')[:50]}..., MongoDB ID: {doc.get('_id')}")
        else:
            logger.warning("❌ MongoDB 会议转录数据为空。")

        logger.info(f"--- 验证 MongoDB 集合 'users' ---")
        # 使用 get_all_users 方法来获取用户数据，它会处理 ObjectId 转换
        users = await mongodb_manager.get_all_users()
        user_count = len(users)
        logger.info(f"MongoDB 集合 'users' 中有 {user_count} 条文档。")
        if user_count > 0:
            logger.info("✅ MongoDB 用户数据存在。")
            for i, user in enumerate(users[:3]):
                logger.info(f"  示例用户 {i+1}: User ID: {user.user_id}, Username: {user.username}, Role: {user.role.value}, MongoDB ID: {user.user_id}") # MongoDB ID 打印 user_id
        else:
            logger.warning("❌ MongoDB 用户数据为空。")

        # 2. 初始化声纹 MilvusManager
        logger.info(f"--- 验证 Milvus 集合 '{settings.MILVUS_VOICE_COLLECTION_NAME}' (声纹) ---")
        voice_milvus_manager = MilvusManager(
            config=settings,
            collection_name=settings.MILVUS_VOICE_COLLECTION_NAME,
            schema_fields=settings.MILVUS_VOICE_SCHEMA_FIELDS
        )
        await voice_milvus_manager.connect(overwrite_collection=False) # 不覆盖，只连接
        if voice_milvus_manager.collection:
            await asyncio.to_thread(voice_milvus_manager.collection.load) # 确保加载到内存
            voice_entity_count = voice_milvus_manager.collection.num_entities 
            logger.info(f"Milvus 集合 '{settings.MILVUS_VOICE_COLLECTION_NAME}' 中有 {voice_entity_count} 个实体。")
            if voice_entity_count > 0:
                logger.info("✅ Milvus 声纹数据存在。")
                sample_voice_data = await voice_milvus_manager.get_all_data(output_fields=["id", "user_name", "role"])
                for i, data in enumerate(sample_voice_data[:3]):
                    logger.info(f"  示例声纹 {i+1}: ID: {data.get('id')}, User: {data.get('user_name')}, Role: {data.get('role')}")
            else:
                logger.warning("❌ Milvus 声纹数据为空。")
        else:
            logger.error(f"❌ 无法连接到 Milvus 集合 '{settings.MILVUS_VOICE_COLLECTION_NAME}'。")


        # 3. 初始化会议文本嵌入 MilvusManager
        logger.info(f"--- 验证 Milvus 集合 '{settings.MILVUS_MEETING_COLLECTION_NAME}' (会议文本嵌入) ---")
        meeting_milvus_manager = MilvusManager(
            config=settings,
            collection_name=settings.MILVUS_MEETING_COLLECTION_NAME,
            schema_fields=settings.MILVUS_MEETING_SCHEMA_FIELDS
        )
        await meeting_milvus_manager.connect(overwrite_collection=False) # 不覆盖，只连接
        if meeting_milvus_manager.collection:
            await asyncio.to_thread(meeting_milvus_manager.collection.load) # 确保加载到内存
            meeting_entity_count = meeting_milvus_manager.collection.num_entities 
            logger.info(f"Milvus 集合 '{settings.MILVUS_MEETING_COLLECTION_NAME}' 中有 {meeting_entity_count} 个实体。")
            if meeting_entity_count > 0:
                logger.info("✅ Milvus 会议文本嵌入数据存在。")
                sample_meeting_data = await meeting_milvus_manager.get_all_data(output_fields=["id", "mongo_id"])
                for i, data in enumerate(sample_meeting_data[:3]):
                    logger.info(f"  示例会议嵌入 {i+1}: Milvus ID: {data.get('id')}, MongoDB ID: {data.get('mongo_id')}")
            else:
                logger.warning("❌ Milvus 会议文本嵌入数据为空。")
        else:
            logger.error(f"❌ 无法连接到 Milvus 集合 '{settings.MILVUS_MEETING_COLLECTION_NAME}'。")

    except Exception as e:
        logger.critical(f"验证过程中发生严重错误: {e}", exc_info=True)
    finally:
        # 关闭所有管理器
        if voice_milvus_manager:
            await voice_milvus_manager.close()
        if meeting_milvus_manager:
            await meeting_milvus_manager.close()
        if mongodb_manager:
            await mongodb_manager.close()
        logger.info("--- 数据库验证完成 ---")

if __name__ == "__main__":
    asyncio.run(verify_database_data())


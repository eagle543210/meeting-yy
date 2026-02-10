import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import logging

# 假设你的 MongoDB 连接 URI 如下，请确保它是正确的
MONGO_URI = "mongodb://localhost:27017/" # 请根据你的实际情况修改
DB_NAME = "meeting_db"
COLLECTION_NAME = "users"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_mongo_connection():
    """
    测试 MongoDB 连接并获取用户数据。
    """
    client = None
    try:
        logging.info("正在连接到 MongoDB...")
        client = AsyncIOMotorClient(MONGO_URI)
        
        # 验证连接是否成功
        await client.admin.command('ping')
        logging.info("🎉 MongoDB 连接成功！")
        
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        logging.info(f"正在查询 '{COLLECTION_NAME}' 集合中的所有文档...")
        
        users_cursor = collection.find({})
        users_list = await users_cursor.to_list(length=None)
        
        if users_list:
            logging.info(f"成功获取 {len(users_list)} 条用户数据。")
            for user in users_list:
                logging.info(f"用户数据: {user}")
        else:
            logging.warning("没有找到任何用户数据。集合可能为空。")
        
    except Exception as e:
        logging.error(f"❌ 连接或查询 MongoDB 时发生错误: {e}")
    finally:
        if client:
            client.close()
            logging.info("MongoDB 连接已关闭。")

if __name__ == "__main__":
    asyncio.run(test_mongo_connection())


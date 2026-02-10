# M:\meeting\services\mongodb_manager.py

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase # 异步 MongoDB 驱动
from typing import Optional, List, Dict, Any # 类型提示
from datetime import datetime, timezone
from bson import ObjectId # 用于处理 MongoDB 的 ObjectId (仅用于 Milvus 的 mongo_id 关联，或作为通用类型提示)
import pytz # 新增：用于时区转换

# 导入配置和模型 (现在直接从 models 包导入)
from config.settings import settings
from models import TranscriptEntry, User, UserRole # 不再导入 PydanticObjectId

logger = logging.getLogger(__name__)

# 定义本地时区，例如 'Asia/Shanghai'
# 你可以根据实际情况修改
LOCAL_TIMEZONE = pytz.timezone('Asia/Shanghai')

class MongoDBManager:
    """
    MongoDBManager 负责管理与 MongoDB 数据库的连接和基本操作。
    它使用 Motor 异步驱动，以支持 FastAPI 的异步特性。
    """
    def __init__(self, host: str, port: int, db_name: str):
        """
        初始化 MongoDBManager。
        Args:
            host (str): MongoDB 主机地址。
            port (int): MongoDB 端口。
            db_name (str): 要连接的数据库名称。
        """
        self.host = host
        self.port = port
        self.db_name = db_name
        self.client: Optional[AsyncIOMotorClient] = None # 异步 MongoDB 客户端
        self.db: Optional[AsyncIOMotorDatabase] = None # 异步 MongoDB 数据库对象
        self.is_connected = False # 连接状态标记

        logger.info(f"MongoDBManager 初始化，目标: {self.host}:{self.port}/{self.db_name}")

    async def connect(self):
        """
        异步连接到 MongoDB 数据库。
        """
        if self.is_connected and self.client is not None and self.db is not None:
            logger.info("MongoDB 数据库已连接，跳过重复连接。")
            return

        try:
            # 使用 settings.MONGO_URI 来连接，更加灵活
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db = self.client[self.db_name]
            # 尝试执行一个简单的操作来验证连接
            await self.db.command('ping') 
            self.is_connected = True
            logger.info("✅ MongoDB 数据库连接成功。")
        except Exception as e:
            logger.critical(f"❌ MongoDB 数据库连接失败: {e}", exc_info=True)
            self.is_connected = False
            self.client = None
            self.db = None
            raise # 重新抛出异常，阻止应用启动

    async def close(self):
        """
        异步关闭 MongoDB 数据库连接。
        """
        if self.client is not None:
            self.client.close()
            self.is_connected = False
            logger.info("✅ MongoDB 数据库连接已关闭。")

    async def save_transcript_entry(self, entry: TranscriptEntry) -> Optional[str]: # 返回类型改为 str
        """
        保存单个转录条目到 MongoDB。
        Args:
            entry (TranscriptEntry): 要保存的转录条目对象。
        Returns:
            Optional[str]: 插入文档的 MongoDB _id (字符串形式)，如果插入失败则为 None。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法保存转录条目。")
            return None
        try:
            collection = self.db[settings.MONGO_TRANSCRIPT_COLLECTION_NAME]
            # model_dump(by_alias=True) 会将 Pydantic 的 id 字段映射为字典中的 _id
            # exclude_none=True 确保如果 id 是 None，则 _id 不会出现在字典中
            doc = entry.model_dump(mode='json', by_alias=True, exclude_none=True) 
            
            result = await collection.insert_one(doc)
            logger.debug(f"转录条目已保存，ID: {result.inserted_id}")
            # 对于 TranscriptEntry，_id 是 UUID 字符串，直接返回
            return str(result.inserted_id) # 确保返回的是字符串
        except Exception as e:
            logger.error(f"保存转录条目失败: {e}", exc_info=True)
            return None

    async def get_all_transcripts_for_meeting(self, meeting_id: str) -> List[TranscriptEntry]:
        """
        获取指定会议的所有转录条目。
        Args:
            meeting_id (str): 会议的 ID。
        Returns:
            List[TranscriptEntry]: 转录条目列表。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法获取转录条目。")
            return []
        try:
            collection = self.db[settings.MONGO_TRANSCRIPT_COLLECTION_NAME]
            cursor = collection.find({"meeting_id": meeting_id}).sort("timestamp", 1)
            docs = await cursor.to_list(length=None) 
            # TranscriptEntry 的 id 字段是 str，alias="_id"，直接传入 doc 即可
            return [TranscriptEntry(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"获取会议 '{meeting_id}' 的所有转录条目失败: {e}", exc_info=True)
            return []

    async def get_documents_by_ids(self, ids: List[str]) -> List[TranscriptEntry]:
        """
        [新增] 根据文档的字符串ID列表获取文档。
        Args:
            ids (List[str]): 文档的 _id 字符串列表。
        Returns:
            List[TranscriptEntry]: 匹配到的转录条目列表。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法根据ID获取文档。")
            return []
        try:
            collection = self.db[settings.MONGO_TRANSCRIPT_COLLECTION_NAME]
            # 使用 $in 操作符查询 _id 在给定列表中的所有文档
            cursor = collection.find({"_id": {"$in": ids}})
            docs = await cursor.to_list(length=None)
            return [TranscriptEntry(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"根据ID列表获取文档失败: {e}", exc_info=True)
            return []
            
    async def get_user(self, user_id: str) -> Optional[User]:
        """
        根据 user_id 从 MongoDB 获取用户信息。
        Args:
            user_id (str): 用户的唯一 ID。
        Returns:
            Optional[User]: 用户对象，如果未找到则为 None。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法获取用户信息。")
            return None
        try:
            collection = self.db["users"] # 用户集合通常名为 'users'
            # User 的 _id 就是 user_id，所以直接用 _id 字段查询
            doc = await collection.find_one({"_id": user_id}) 
            if doc:
                # User 的 user_id 字段是 str，alias="_id"，直接传入 doc 即可
                return User(**doc)
            return None
        except Exception as e:
            logger.error(f"获取用户 '{user_id}' 信息失败: {e}", exc_info=True)
            return None

    async def add_or_update_user(self, user: User) -> bool:
        """
        添加或更新用户信息到 MongoDB。
        Args:
            user (User): 要添加或更新的用户对象。
        Returns:
            bool: 如果操作成功则为 True，否则为 False。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法添加或更新用户。")
            return False
        try:
            collection = self.db["users"]
            
            # 获取 Pydantic 模型的字典表示，by_alias=True 确保 username 字段被转换为 name
            # exclude_none=True 排除值为 None 的字段
            doc_data_for_set = user.model_dump(
                mode='json', 
                by_alias=True, 
                exclude_none=True,
                exclude={'user_id'} # 排除 user_id，因为它作为 _id 存在于查询条件中，不能在 $set 中修改
            )
            
            # query_filter 使用 user_id，它在 User 模型中被 alias 为 _id
            query_filter = {"_id": user.user_id} 

            result = await collection.update_one(
                query_filter, 
                {"$set": doc_data_for_set},
                upsert=True # 如果不存在则插入
            )
            logger.info(f"用户 '{user.username}' (ID: {user.user_id}) 已在 MongoDB 中注册/更新。")
            return True
        except Exception as e:
            logger.error(f"添加或更新用户 '{getattr(user, 'user_id', 'Unknown ID')}' 失败: {e}", exc_info=True)
            return False

    async def update_user_role(
        self, 
        user_id: str, 
        new_role: UserRole,
        new_name: Optional[str] = None  
    ) -> bool:
        """
        更新指定用户的角色和名称。
        Args:
            user_id (str): 用户的唯一 ID
            new_role (UserRole): 用户的新角色
            new_name (Optional[str]): 用户的新名称（可选）
        Returns:
            bool: 更新成功返回True，否则False
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法更新用户。")
            return False
            
        try:
            collection = self.db["users"]
            update_data = {
                "role": new_role.value,
                # 使用 timezone.utc 确保 last_active 总是以正确的时区感知对象存储
                "last_active": datetime.now(timezone.utc)
            }
            
            # 如果提供了新名称，则添加到更新数据
            if new_name is not None:
                update_data["name"] = new_name
            
            result = await collection.update_one(
                {"_id": user_id},
                {"$set": update_data}
            )
            
            if result.matched_count > 0:
                log_msg = f"用户 '{user_id}' 已更新: 角色={new_role.value}"
                if new_name:
                    log_msg += f", 名称={new_name}"
                logger.info(log_msg)
                return True
            else:
                logger.warning(f"未找到用户 '{user_id}'")
                return False
                
        except Exception as e:
            logger.error(f"更新用户 '{user_id}' 失败: {e}", exc_info=True)
            return False

    async def get_all_users(self) -> List[User]:
        """
        获取所有注册用户的信息，并将 UTC 时间转换为本地时区。
        Returns:
            List[User]: 所有用户对象的列表。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法获取所有用户。")
            return []
        try:
            collection = self.db["users"]
            cursor = collection.find({})
            users_data = await cursor.to_list(length=None)
            
            processed_users = []
            for user_doc in users_data:
                for key in ['created_at', 'last_active']:
                    if key in user_doc and isinstance(user_doc[key], str):
                        try:
                            # 尝试解析带时区信息的 ISO 格式字符串
                            dt_obj = datetime.fromisoformat(user_doc[key])
                            # 如果解析出的对象是天真的（naive），则假定它为 UTC
                            if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
                                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                            
                            user_doc[key] = dt_obj.astimezone(LOCAL_TIMEZONE)
                        except ValueError:
                            logger.warning(f"字段 '{key}' 的值 '{user_doc[key]}' 格式无效，跳过转换。")
                    elif key in user_doc and isinstance(user_doc[key], datetime):
                        # 如果是 datetime 对象，则直接进行时区转换
                        # 如果是天真的（naive）datetime 对象，则假定它为 UTC
                        if user_doc[key].tzinfo is None or user_doc[key].tzinfo.utcoffset(user_doc[key]) is None:
                            user_doc[key] = user_doc[key].replace(tzinfo=timezone.utc)
                        user_doc[key] = user_doc[key].astimezone(LOCAL_TIMEZONE)
            
                processed_users.append(User(**user_doc))

            return processed_users

        except Exception as e:
            logger.error(f"获取所有用户失败: {e}", exc_info=True)
            return []

    async def get_distinct_meeting_ids(self) -> List[str]:
        """
        获取所有会议转录中不重复的 meeting_id。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法获取不重复的会议ID。")
            return []
        try:
            collection = self.db[settings.MONGO_TRANSCRIPT_COLLECTION_NAME]
            meeting_ids = await collection.distinct("meeting_id")
            return meeting_ids
        except Exception as e:
            logger.error(f"获取不重复的会议ID失败: {e}", exc_info=True)
            return []

    async def get_distinct_speakers(self) -> List[str]:
        """
        获取所有会议转录中不重复的 speaker_id (即 username)。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法获取不重复的发言人。")
            return []
        try:
            collection = self.db[settings.MONGO_TRANSCRIPT_COLLECTION_NAME]
            speakers = await collection.distinct("speaker_id")
            return speakers
        except Exception as e:
            logger.error(f"获取不重复的发言人失败: {e}", exc_info=True)
            return []

    async def get_user_speeches_for_meeting(self, meeting_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取指定会议中特定用户或所有用户的发言内容。
        Args:
            meeting_id (str): 会议的 ID。
            user_id (Optional[str]): 可选的用户 ID，如果提供则只返回该用户的发言。
        Returns:
            List[Dict[str, Any]]: 符合条件的发言列表。
        """
        if self.db is None:
            logger.error("MongoDB 数据库未连接，无法获取用户发言。")
            return []
        try:
            collection = self.db[settings.MONGO_TRANSCRIPT_COLLECTION_NAME]
            query = {"meeting_id": meeting_id}
            if user_id:
                query["user_id"] = user_id
            
            cursor = collection.find(query).sort("timestamp", 1)
            docs = await cursor.to_list(length=None)
            # 对于返回原始字典的场景，直接返回即可，无需特殊 _id 转换
            return docs
        except Exception as e:
            logger.error(f"获取会议 '{meeting_id}' 的用户发言失败 (用户: {user_id}): {e}", exc_info=True)
            return []

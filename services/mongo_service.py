# services\mongo_service.py
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase 
from models.user import User
from models.role import UserRole 
from config.settings import settings 

logger = logging.getLogger(__name__)

class MongoService:
    """
    封装所有与 MongoDB 交互的逻辑。
    负责用户管理、实时转录数据存储和查询。
    """
    def __init__(self):
        logger.info("MongoService 初始化中...")
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None 
        self.mongo_uri: str = settings.MONGO_URI
        self.database_name: str = settings.MONGO_DB_NAME
        logger.info(f"MongoService 已配置，URI: {self.mongo_uri}, DB Name: {self.database_name}")

    async def connect(self):
        if self.client and self.db:
            logger.info("MongoDB 客户端已连接，跳过重复连接。")
            return self.db
            
        try:
            logger.info(f"正在连接到 MongoDB: {self.mongo_uri}...")
            self.client = AsyncIOMotorClient(self.mongo_uri)
            self.db = self.client[self.database_name]
            
            # 尝试执行一个简单的操作来验证连接，例如列出集合
            # await self.db.command("ping") # 可以添加这一行进行更严格的连接验证

            # === 修复点：确保 User.set_db_instance 在连接成功后被调用 ===
            User.set_db_instance(self.db)
            logger.info("MongoDB 数据库实例已成功设置到 User 模型。")

            logger.info(f"MongoService 成功连接到 MongoDB。URI: {self.mongo_uri}, DB: {self.database_name}")
            logger.info(f"MongoService.client 类型: {type(self.client)}")
            logger.info(f"MongoService.db 类型: {type(self.db)}")
            
        except Exception as e:
            logger.error(f"MongoService 连接到 MongoDB 失败: {e}", exc_info=True)
            self.client = None # 连接失败时确保清空
            self.db = None     # 连接失败时确保清空
            raise # 确保异常被重新抛出，以便上层调用者（BackendCoordinator）能够捕获并停止应用启动

    def close(self):
        """关闭 MongoDB 连接。"""
        if self.client:
            self.client.close()
            self.client = None 
            self.db = None     
            logger.info("MongoDB 连接已关闭。")

    
     async def get_user_by_voiceprint_id(self, voiceprint_id: str) -> Optional[User]:
        """
        根据用户的声纹ID（通常与用户ID相同）从 MongoDB 中获取用户数据。
        直接使用 User 模型的方法，该方法现在应该已通过 set_db_instance 获得数据库实例。
        """
        try:
            # 确保 User 模型已正确设置数据库实例
            user = await User.find_by_voiceprint(voiceprint_id)
            if user:
                logger.debug(f"通过声纹ID '{voiceprint_id}' 找到用户: {user.username}")
            else:
                logger.warning(f"未通过声纹ID '{voiceprint_id}' 找到用户。")
            return user # 返回 User 模型实例
        except RuntimeError as e:
            logger.error(f"User 模型数据库实例未设置，无法获取用户数据 (声纹ID: {voiceprint_id}): {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"从 MongoDB 获取用户数据失败 (声纹ID: {voiceprint_id}): {e}", exc_info=True)
            return None

    async def find_user_by_id(self, user_id: str) -> Optional[User]:
        """
        根据 user_id 从 MongoDB 中查找用户。
        直接使用 User 模型的方法。
        """
        try:
            user = await User.find_by_id(user_id)
            return user # 返回 User 模型实例
        except RuntimeError as e:
            logger.error(f"User 模型数据库实例未设置，无法通过 ID '{user_id}' 查找用户: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"通过 ID '{user_id}' 查找用户失败: {e}", exc_info=True)
            return None
    async def save_user(self, user_instance: User) -> Dict[str, Any]:
        """
        保存或更新用户数据到 MongoDB。
        直接调用 User 模型实例的 save 方法。
        """
        if not isinstance(user_instance, User):
            logger.error("save_user 期望一个 User 模型实例。")
            return {"status": "error", "message": "期望 User 模型实例。"}
        
        try:
            result = await user_instance.save()
            return {"status": "success", "user_id": user_instance.user_id, **result}
        except RuntimeError as e:
            logger.error(f"User 模型数据库实例未设置，无法保存用户数据: {e}", exc_info=True)
            return {"status": "error", "message": f"MongoDB 服务未就绪: {e}"}
        except Exception as e:
            logger.error(f"保存用户 '{user_instance.username}' 数据失败: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    
    async def find_all_users(self) -> List[User]:
        """
        查找所有用户。
        直接使用 User 模型的方法。
        """
        logger.debug("正在查找所有用户...")
        try:
            return await User.find_all()
        except RuntimeError as e:
            logger.error(f"User 模型数据库实例未设置，无法查找所有用户: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"查找所有用户失败: {e}", exc_info=True)
            return []
            
    
    async def get_meeting_full_transcript(self, meeting_id: str) -> List[Dict[str, Any]]:
        
        try:
            transcripts_collection = self.db.get_collection("transcripts") 
            cursor = transcripts_collection.find({"meeting_id": meeting_id}).sort("timestamp", 1)
            full_transcript = await cursor.to_list(length=None)
            logger.debug(f"获取会议 {meeting_id} 的 {len(full_transcript)} 个转录块。")
            return full_transcript
        except Exception as e:
            logger.error(f"获取会议 '{meeting_id}' 的完整转录失败: {e}", exc_info=True)
            return []

    async def get_user_speeches_in_meeting(self, meeting_id: str, user_id: str) -> List[Dict[str, Any]]:
       
        try:
            transcripts_collection = self.db.get_collection("transcripts") 
            cursor = transcripts_collection.find({"meeting_id": meeting_id, "user_id": user_id}).sort("timestamp", 1)
            user_speeches = await cursor.to_list(length=None)
            logger.debug(f"获取会议 {meeting_id} 中用户 {user_id} 的 {len(user_speeches)} 个发言。")
            return user_speeches
        except Exception as e:
            logger.error(f"获取用户 '{user_id}' 在会议 '{meeting_id}' 中的发言失败: {e}", exc_info=True)
            return []

    async def save_realtime_transcript_chunk(self, meeting_id: str, client_id: str, user_id: str, user_name: str, role: str, transcript: str, stt_confidence: float, speaker_confidence: float) -> Dict[str, Any]:
       
        try:
            transcripts_collection = self.db.get_collection("transcripts") 
            
            transcript_chunk_data = {
                "meeting_id": meeting_id,
                "client_id": client_id,
                "user_id": user_id,
                "user_name": user_name,
                "role": role,
                "transcript": transcript,
                "stt_confidence": stt_confidence,
                "speaker_confidence": speaker_confidence,
                "timestamp": datetime.utcnow()
            }
            
            result = await transcripts_collection.insert_one(transcript_chunk_data)
            logger.debug(f"实时转录块插入成功，ID: {result.inserted_id}")
            return {"status": "success", "inserted_id": str(result.inserted_id)}
        except Exception as e:
            logger.error(f"保存实时转录块失败: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

# M:\meeting\models.py

from datetime import datetime
from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema
from bson import ObjectId # 导入 ObjectId

# ====================================================================
# 自定义 Pydantic ObjectId 类型
# 确保 MongoDB 的 ObjectId 能够被 Pydantic 正确处理为字符串
# ====================================================================
class PydanticObjectId(ObjectId):
    """
    一个自定义的 Pydantic 类型，用于处理 MongoDB 的 ObjectId。
    它确保 ObjectId 在 Pydantic 模型中被视为字符串。
    """
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def validate_from_mongodb_id(value: Any) -> ObjectId:
            # 如果值已经是 ObjectId，直接返回
            if isinstance(value, ObjectId):
                return value
            # 如果是字符串，尝试转换为 ObjectId
            if isinstance(value, str):
                try:
                    return ObjectId(value)
                except Exception as e:
                    raise PydanticCustomError(
                        'invalid_object_id',
                        'Value is not a valid ObjectId string',
                        {'error': str(e)}
                    )
            # 对于其他类型，抛出错误
            raise PydanticCustomError(
                'invalid_object_id_type',
                'Value must be an ObjectId or a string',
                {'value_type': type(value)}
            )

        # 定义 Pydantic 如何解析（验证）和序列化这个类型
        return core_schema.json_or_python_schema(
            # JSON schema: 当从 JSON 解析时，期望是字符串
            json_schema=core_schema.str_schema(), 
            # Python schema: 当从 Python 对象解析时，接受 ObjectId 或字符串，并通过验证器处理
            python_schema=core_schema.union_schema([ 
                core_schema.is_instance_schema(ObjectId), # 直接接受 ObjectId 实例
                core_schema.str_schema(), # 直接接受字符串
                core_schema.no_info_after_validator_function(validate_from_mongodb_id), # 使用我们的验证函数
            ]),
            # 序列化：将 PydanticObjectId 实例转换为字符串
            serialization=core_schema.to_string_ser_schema(
                core_schema.union_schema([
                    core_schema.is_instance_schema(ObjectId), # 序列化 ObjectId 实例
                    core_schema.str_schema(), # 序列化字符串
                ])
            )
        )

    # 定义 ObjectId 实例的字符串表示，这将用于默认的 str() 转换
    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return f"PydanticObjectId('{super().__str__()}')"


# ====================================================================
# Pydantic 模型基类配置
# ====================================================================
class BaseMongoModel(BaseModel):
    """
    所有 MongoDB 模型的基础类，处理 _id 字段的映射和类型转换。
    """
    # MongoDB 的 _id 字段，在 Pydantic 中映射为 id
    # 现在使用自定义的 PydanticObjectId 类型
    id: Optional[PydanticObjectId] = Field(alias="_id", default=None, description="MongoDB 文档的唯一 ID")

    # 配置 Pydantic 模型
    model_config = ConfigDict(
        populate_by_name=True,  # 允许通过字段名或别名赋值
        arbitrary_types_allowed=True, # 允许任意类型，这在处理 ObjectId 等自定义类型时有用
        json_encoders={ObjectId: str} # 将 ObjectId 编码为字符串，用于 JSON 序列化 (兼容旧版本和某些场景)
    )

# ====================================================================
# 用户角色枚举
# ====================================================================
class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    TEAM_LEAD = "team_lead"
    MEMBER = "member"
    GUEST = "guest"
    HOST = "host"

# ====================================================================
# 权限枚举
# ====================================================================
class Permission(str, Enum):
    EDIT_ROLES = "edit_roles"
    EXPORT_MINUTES = "export_minutes"
    EXPORT_USER_SPEECH = "export_user_speech"
    GENERATE_REPORTS = "generate_reports"
    MANAGE_MEETINGS = "manage_meetings"

# ====================================================================
# 定义具体的数据模型
# ====================================================================

# 用户模型
class User(BaseMongoModel):
    user_id: str = Field(..., description="用户在系统中的唯一标识符，通常与声纹 ID 关联")
    username: str = Field(..., description="用户的显示名称")
    role: UserRole = Field(UserRole.GUEST, description="用户的角色")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="用户创建时间")
    last_active: datetime = Field(default_factory=datetime.utcnow, description="用户最后活跃时间")

# 转录条目模型
class TranscriptEntry(BaseMongoModel):
    meeting_id: str = Field(..., description="所属会议的唯一 ID")
    client_id: str = Field(..., description="客户端在语音流中的唯一标识符")
    user_id: str = Field(..., description="说话人对应的用户 ID")
    speaker_id: str = Field(..., description="说话人的名称（如：张三）")
    role: str = Field(..., description="发言人在此会议中的角色")
    text: str = Field(..., description="转录的文本内容")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="发言的时间戳")
    
    # 嵌入字段 (可选，因为嵌入可能在稍后生成)
    embedding: Optional[List[float]] = Field(None, description="文本内容的向量嵌入")

# 会议模型 (用于存储会议元数据、总结、行动项等)
class Meeting(BaseMongoModel):
    meeting_id: str = Field(..., description="会议的唯一 ID")
    title: str = Field(..., description="会议标题")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="会议开始时间")
    end_time: Optional[datetime] = Field(None, description="会议结束时间")
    participants: List[str] = Field(default_factory=list, description="参与者用户ID列表") # 存储 user_id
    summary: Optional[str] = Field(None, description="会议总结")
    action_items: Optional[List[str]] = Field(default_factory=list, description="会议行动项")
    decisions: Optional[List[str]] = Field(default_factory=list, description="会议决策")

# 角色权限模型 (用于存储每个角色对应的权限列表)
class RolePermission(BaseMongoModel):
    role: UserRole = Field(..., description="用户角色")
    permissions: List[Permission] = Field(default_factory=list, description="该角色拥有的权限列表")


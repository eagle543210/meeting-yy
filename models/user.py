# models\user.py

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

# 修正：从 .role 导入 UserRole (因为 UserRole 现在只在 models/role.py 中定义)
from .role import UserRole 

class User(BaseModel):
    """
    用户数据模型，与 MongoDB 用户集合对应。
    继承 Pydantic BaseModel 以获得数据验证和序列化能力。
    """
    user_id: str = Field(..., alias="_id", description="用户的唯一标识符，通常与声纹ID关联") # MongoDB 的 _id 字段
    username: str = Field(..., alias="name", description="用户的友好名称") # 保持与原 name 字段的兼容性
    role: UserRole = Field(UserRole.GUEST, description="用户的角色") # 直接使用 UserRole 枚举
    created_at: datetime = Field(default_factory=datetime.utcnow, description="用户创建时间 (UTC)")
    last_active: datetime = Field(default_factory=datetime.utcnow, description="用户最后活跃时间 (UTC)")

    class Config:
        populate_by_name = True # 允许使用字段别名来赋值
        json_encoders = {
            datetime: lambda dt: dt.isoformat(), # 序列化 datetime 为 ISO 格式字符串
            UserRole: lambda role: role.value # 序列化 UserRole 枚举为字符串值
        }
        arbitrary_types_allowed = True # 允许在模型中使用非 Pydantic 类型，如 ObjectId (如果直接从 PyMongo 查询结果构建)

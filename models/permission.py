# models\permission.py

from enum import Enum
from typing import List
from pydantic import BaseModel, Field

from .role import UserRole 

# 定义权限枚举
class Permission(str, Enum):
    EDIT_ROLES = "edit_roles"
    EXPORT_MINUTES = "export_minutes"
    EXPORT_USER_SPEECH = "export_user_speech"
    GENERATE_REPORTS = "generate_reports"
    MANAGE_MEETINGS = "manage_meetings"

# 角色权限模型 (用于存储每个角色对应的权限列表)
class RolePermission(BaseModel): # 注意：这里不再继承 BaseMongoModel，因为它没有 _id 字段
    role: UserRole = Field(..., description="用户角色")
    permissions: List[Permission] = Field(default_factory=list, description="该角色拥有的权限列表")

    class Config:
        populate_by_name = True # 允许使用字段别名来赋值
        json_encoders = {
            UserRole: lambda role: role.value # 序列化 UserRole 枚举为字符串值
        }
        arbitrary_types_allowed = True

# models\role.py

from enum import Enum

class UserRole(str, Enum): # 继承 str 和 Enum，允许枚举成员为字符串值
    ADMIN = "ADMIN"         # 管理员 (改为大写)
    MANAGER = "MANAGER"     # 经理 (改为大写)
    TEAM_LEAD = "TEAM_LEAD"# 团队领导 (改为大写)
    MEMBER = "MEMBER"       # 成员 (改为大写)
    GUEST = "GUEST"         # 访客 (改为大写)
    REGISTERED_USER = "REGISTERED_USER" # 用于 Milvus 中已注册的用户 (改为大写)
    UNKNOWN = "UNKNOWN"     # 用于无法识别的用户 (改为大写)
    ERROR = "ERROR"         # 用于声纹识别服务出错的情况 (改为大写)
    HOST = "HOST" # 确保 HOST 角色存在，与模拟数据一致
    CLIENT = "CLIENT" # 确保 CLIENT 角色存在，与模拟数据一致

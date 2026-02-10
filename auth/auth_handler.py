# M:\meeting\auth\auth_handler.py
import time
import os
import logging
from typing import Dict, Any, Optional

import jwt # pip install python-jose[cryptography] 或 pip install PyJWT
from jwt import PyJWTError # 导入 PyJWTError 用于更具体的错误处理

from config.settings import settings # 导入配置，可以从这里获取 JWT 密钥

logger = logging.getLogger(__name__)

# 从环境中获取 JWT 密钥，如果不存在则使用一个默认值 (生产环境强烈不推荐默认值)
# 更好的做法是在 settings.py 中定义并从环境变量加载
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-please-change-this") 
JWT_ALGORITHM = "HS256" # 使用 HS256 算法

# 建议：将 JWT 密钥和算法定义在 settings.py 中
# from config.settings import settings
# JWT_SECRET = settings.JWT_SECRET_KEY
# JWT_ALGORITHM = settings.JWT_ALGORITHM

def signJWT(user_id: str, user_role: str) -> Dict[str, str]:
    """
    使用 JWT 签名用户的 ID 和角色，生成访问令牌。
    """
    payload = {
        "user_id": user_id,
        "user_role": user_role, # 存储用户角色
        "expires": time.time() + (60 * 60 * 24) # 令牌有效期设置为 24 小时
    }
    try:
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        logger.info(f"为用户 '{user_id}' (角色: {user_role}) 生成 JWT 令牌。")
        return {"access_token": token}
    except Exception as e:
        logger.error(f"生成 JWT 令牌失败: {e}", exc_info=True)
        return {"error": "Failed to generate token"}

def decodeJWT(token: str) -> Optional[Dict[str, Any]]:
    """
    解码并验证 JWT 令牌。
    """
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        # 检查令牌是否过期
        if decoded_token.get("expires") and decoded_token["expires"] >= time.time():
            logger.debug(f"JWT 令牌验证成功，用户 ID: {decoded_token.get('user_id')}")
            return decoded_token
        else:
            logger.warning(f"JWT 令牌已过期或缺少过期时间。用户 ID: {decoded_token.get('user_id')}")
            return None # 令牌过期
    except PyJWTError as e:
        logger.warning(f"JWT 令牌解码或验证失败: {e}", exc_info=True)
        return None # 无效令牌
    except Exception as e:
        logger.error(f"JWT 解码时发生未知错误: {e}", exc_info=True)
        return None
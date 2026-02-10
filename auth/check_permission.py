# auth/check_permission.py
import functools
import logging
from typing import Callable, Any, Awaitable, Optional

# 假设你正在使用 FastAPI，你需要导入 Request
# 如果是其他框架（如 Flask），导入其对应的请求对象
from fastapi import Request, HTTPException, status # 导入 HTTPException 和 status 用于错误处理

# 从你的 models 模块导入，确保路径正确
from models.user import User # 假设你有一个 User 模型
from models.role import UserRole # 导入 UserRole 枚举
from models.permission import Permission # 导入 Permission 枚举

logger = logging.getLogger(__name__)

# 权限与角色映射 (与之前保持一致)
ROLE_PERMISSIONS = {
    UserRole.ADMIN: {
        Permission.EDIT_ROLES,
        Permission.EXPORT_MINUTES,
        Permission.EXPORT_USER_SPEECH,
        Permission.GENERATE_REPORTS
    },
    UserRole.MANAGER: {
        Permission.EXPORT_MINUTES,
        Permission.EXPORT_USER_SPEECH,
        Permission.GENERATE_REPORTS
    },
    UserRole.TEAM_LEAD: {
        Permission.EXPORT_MINUTES,
        Permission.EXPORT_USER_SPEECH
    },
    UserRole.MEMBER: {
        Permission.EXPORT_USER_SPEECH
    },
    UserRole.GUEST: set()
}

async def get_current_user(request: Request) -> User:
    """
    从请求中获取当前认证用户。
    
    这通常涉及：
    1. 从 Authorization 请求头解析 JWT 令牌。
    2. 验证令牌的有效性（签名、过期时间等）。
    3. 从令牌负载中提取用户 ID。
    4. 根据用户 ID 从数据库加载完整的用户对象（包括角色信息）。
    
    Args:
        request (Request): FastAPI 的请求对象。
        
    Returns:
        User: 当前认证的用户对象。
        
    Raises:
        HTTPException: 如果认证失败、令牌无效或用户不存在。
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        logger.warning("认证失败: 请求头中缺少 Authorization。")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证凭据缺失",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 检查是否为 Bearer Token 格式
    scheme, token = auth_header.split(" ", 1)
    if scheme.lower() != "bearer":
        logger.warning(f"认证失败: 未知认证方案 '{scheme}'。")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="不支持的认证方案",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # --- 实际的 JWT 验证和用户查找逻辑会在这里 ---
    # ⚠️ 警告: 以下是模拟逻辑，生产环境请务必替换为真实的 JWT 验证和数据库查询
    logger.warning("get_current_user: ⚠️ 正在使用模拟的用户和令牌验证。请在生产环境替换此逻辑！")
    
    # 模拟 JWT 解码，从令牌中获取 user_id
    simulated_user_id = ""
    if token == "admin_jwt_token":
        simulated_user_id = "admin_user_123"
    elif token == "manager_jwt_token":
        simulated_user_id = "manager_user_456"
    elif token == "member_jwt_token":
        simulated_user_id = "member_user_789"
    else:
        logger.warning(f"认证失败: 无效或未知的令牌 '{token}'。")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 根据 user_id 从数据库加载用户
    # 假设 User.find_by_id 是一个异步方法，用于从数据库查找用户
    user = await User.find_by_id(simulated_user_id)
    
    if not user:
        logger.warning(f"认证失败: 令牌有效但用户 '{simulated_user_id}' 不存在。")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    logger.info(f"用户 '{user.username}' (ID: {user.user_id}, 角色: {user.role.value}) 已认证。")
    return user

def check_permission(required_permission: Permission) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """
    一个权限检查装饰器，用于保护异步函数。
    它通过 `get_current_user` 获取当前用户，并根据用户的角色检查权限。
    
    Args:
        required_permission (Permission): 调用此函数所需的具体权限。
        
    Returns:
        Callable: 装饰器函数。
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 在 FastAPI 中，如果 get_current_user 是一个依赖（Dependency），
            # 那么它会将返回的用户对象作为参数注入到被装饰的函数中。
            # 我们可以通过 kwargs 获取这个注入的用户对象。
            # 假设注入的用户对象名为 `current_user`。
            current_user: Optional[User] = kwargs.get("current_user")

            # 如果 current_user 没有通过依赖注入传入，尝试手动获取
            # 这通常发生在直接调用 backend.py 中的方法而非通过 FastAPI 路由时
            if not current_user:
                # 尝试从 args 中找到 Request 对象并获取用户
                request_obj: Optional[Request] = None
                for arg in args:
                    if isinstance(arg, Request):
                        request_obj = arg
                        break
                
                if request_obj:
                    try:
                        current_user = await get_current_user(request_obj)
                    except HTTPException as e:
                        raise e # 重新抛出 HTTP 异常
                    except Exception as e:
                        logger.error(f"获取当前用户时发生未知错误: {e}", exc_info=True)
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="内部服务器错误: 无法获取用户身份",
                        )
                else:
                    logger.error("权限检查失败: 无法获取 Request 对象或当前用户上下文。")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="认证上下文缺失或配置错误",
                    )
            
            # 如果依然没有获取到用户，则抛出未授权错误
            if not current_user:
                 logger.error("权限检查失败: 未能识别当前用户。")
                 raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="认证失败: 无法识别您的身份。",
                    headers={"WWW-Authenticate": "Bearer"},
                 )

            current_user_role = current_user.role
            
            # 获取当前角色所拥有的权限集合
            allowed_permissions = ROLE_PERMISSIONS.get(current_user_role, set())
            
            # 检查用户是否拥有所需权限
            if required_permission in allowed_permissions:
                return await func(*args, **kwargs)
            else:
                logger.warning(f"权限不足: 用户 '{current_user.username}' (角色: {current_user_role.value}) 无权执行 '{required_permission.value}'。")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"权限不足: 您没有执行此操作的权限 ({required_permission.value})",
                )
        return wrapper
    return decorator
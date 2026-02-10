# M:\meeting\backend\monitor_manager.py
import logging
from typing import Set, Dict, Any # 确保导入 Dict 和 Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState # 确保导入 WebSocketState

# ⚠️ 移除这里对 system_stats 的导入。
# system_stats 将通过构造函数传入，实现依赖注入，避免状态不同步问题。

logger = logging.getLogger(__name__)

class MonitorManager:
    """
    管理监控 WebSocket 连接，用于广播系统状态。
    """
    def __init__(self, system_stats_ref: Dict[str, Any]): # <--- CRITICAL CHANGE: 添加 system_stats_ref 参数
        # 存储所有活跃的监控 WebSocket 连接
        self.monitor_clients: Set[WebSocket] = set()
        # 存储 system_stats 的引用，以便在类方法中访问和更新全局状态
        self.system_stats = system_stats_ref # <--- CRITICAL CHANGE: 存储引用
        logger.info("MonitorManager 初始化完成。")

    async def connect(self, websocket: WebSocket):
        """
        处理新的监控 WebSocket 连接。
        """
        await websocket.accept()
        self.monitor_clients.add(websocket)
        logger.info("监控客户端连接成功。")
        # 连接成功后立即发送当前系统状态
        await self.send_system_status(websocket)

    def disconnect(self, websocket: WebSocket):
        """
        处理监控 WebSocket 连接断开。
        """
        if websocket in self.monitor_clients:
            self.monitor_clients.remove(websocket)
            logger.info("监控客户端断开连接。")

    async def send_system_status(self, websocket: WebSocket):
        """
        向指定监控客户端发送当前系统状态。
        """
        try:
            # 检查 WebSocket 状态，避免发送到关闭的连接
            if websocket.client_state != WebSocketState.CONNECTED:
                raise WebSocketDisconnect(f"监控 WebSocket 处于非连接状态: {websocket.client_state}")

            await websocket.send_json({
                "type": "system_status",
                "data": self.system_stats, # <--- CHANGE: 使用 self.system_stats
                "timestamp": datetime.now().isoformat()
            })
            logger.debug("已向监控客户端发送系统状态。")
        except (RuntimeError, WebSocketDisconnect) as e:
            logger.warning(f"尝试向已关闭的监控 WebSocket 发送系统状态: {e}. 正在断开连接。")
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"发送系统状态到监控客户端失败: {str(e)}", exc_info=True)
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """
        向所有活跃的监控 WebSocket 连接广播消息。
        """
        disconnected_clients = []
        for client in list(self.monitor_clients): # 迭代副本以安全地删除
            try:
                # 检查 WebSocket 状态，避免发送到关闭的连接
                if client.client_state != WebSocketState.CONNECTED:
                    raise WebSocketDisconnect(f"广播目标监控 WebSocket 处于非连接状态: {client.client_state}")
                await client.send_json(message)
            except (RuntimeError, WebSocketDisconnect) as e:
                logger.warning(f"尝试向已关闭的监控 WebSocket 广播消息: {e}. 正在断开连接。")
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"广播消息到监控客户端失败: {str(e)}", exc_info=True)
                disconnected_clients.append(client)

        # 移除已断开的客户端
        for client in disconnected_clients:
            self.disconnect(client)
            logger.debug("已从监控客户端列表中移除断开连接的客户端。")
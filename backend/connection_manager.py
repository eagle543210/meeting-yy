# M:\meeting\backend\connection_manager.py

import json
import logging
import asyncio
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Set, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# ⚠️ 不再从 app 模块导入 system_stats 以避免循环导入。
# system_stats 将通过 __init__ 方法传入。
from config.settings import settings

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    WebSocket 连接管理器，负责处理客户端连接、断开连接，
    以及向单个客户端或会议中的所有客户端发送消息。
    支持按会议ID进行分组广播和心跳机制。
    """
    def __init__(self, system_stats_ref: Dict[str, Any]):
        """
        初始化 ConnectionManager。
        :param system_stats_ref: 对全局系统统计字典的引用。
        """
        logger.info("初始化 ConnectionManager...")
        # 存储所有活跃的 WebSocket 连接及其会议ID
        # { client_id: (WebSocket, meeting_id) }
        self.active_connections: Dict[str, Tuple[WebSocket, str]] = {}
        
        # 存储每个客户端ID对应的心跳任务
        self.ping_tasks: Dict[str, asyncio.Task] = {}

        # MonitorManager 实例，通过 setter 方法设置
        self._monitor_manager: Optional[Any] = None

        # 对全局系统统计字典的引用
        self.system_stats = system_stats_ref

        logger.info("ConnectionManager 属性已设置。")

    def set_monitor_manager(self, manager: Any):
        """
        设置 MonitorManager 实例，用于广播系统状态更新。
        """
        self._monitor_manager = manager
        logger.info("MonitorManager 已设置到 ConnectionManager。")

    async def connect(self, websocket: WebSocket, client_id: str, meeting_id: str):
        """
        处理新的 WebSocket 连接。
        """
        await websocket.accept()
        self.active_connections[client_id] = (websocket, meeting_id)

        # 启动心跳任务
        ping_task = asyncio.create_task(self.send_ping_periodically(
            client_id, websocket, meeting_id, interval=settings.WEBSOCKET_PING_INTERVAL
        ))
        self.ping_tasks[client_id] = ping_task

        # 更新系统统计
        # total_connections 在这里递增，因为它代表历史总数
        self.system_stats["total_connections"] += 1 
        self.system_stats["active_connections"] = len(self.active_connections)
        # 重新计算活跃会议数，通过遍历 active_connections 获取唯一的 meeting_id
        self.system_stats["meetings_active"] = len(set(mid for _, mid in self.active_connections.values()))

        logger.info(f"客户端 {client_id} 连接成功，会议ID: {meeting_id}")
        logger.debug(f"当前活跃连接总数: {len(self.active_connections)}, 总连接数: {self.system_stats['total_connections']}")

        # 通知监控客户端系统状态更新
        # if self._monitor_manager:
        #     await self._monitor_manager.broadcast({
        #         "type": "system_status_update",
        #         "data": self.system_stats,
        #         "timestamp": datetime.now().isoformat()
        #     })

    async def disconnect(self, client_id: str, meeting_id: Optional[str] = None, expected_websocket: Optional[WebSocket] = None):
        """
        处理 WebSocket 连接断开。
        如果 meeting_id 未提供，将尝试从 active_connections 中查找。
        新增 expected_websocket 参数以防止竞态条件。
        """
        # 首先尝试取消心跳任务
        if client_id in self.ping_tasks:
            self.ping_tasks[client_id].cancel()
            del self.ping_tasks[client_id]
            logger.debug(f"已取消客户端 {client_id} 的心跳任务。")

        # 从 active_connections 中获取连接信息
        connection_info = self.active_connections.get(client_id)
        if connection_info:
            current_websocket, actual_meeting_id = connection_info

            # 防止竞态条件：只有当断开请求对应的websocket与当前活跃的websocket相同时才执行清理
            if expected_websocket and current_websocket != expected_websocket:
                logger.warning(f"Disconnect called for client {client_id}, but the active websocket does not match the expected one. Ignoring cleanup for old socket.")
                return

            # 现在可以安全地移除
            self.active_connections.pop(client_id, None)

            # 如果传入的 meeting_id 为 None，则使用 active_connections 中存储的实际 meeting_id
            if meeting_id is None:
                meeting_id = actual_meeting_id
            
            logger.info(f"客户端 {client_id} (会议 {meeting_id}) 已从活跃连接中移除。")
            
            # 更新系统统计
            self.system_stats["active_connections"] = len(self.active_connections)
            self.system_stats["meetings_active"] = len(set(mid for _, mid in self.active_connections.values()))

            logger.info(f"客户端 {client_id} 彻底断开。")
            logger.debug(f"当前活跃连接总数: {len(self.active_connections)}, 总连接数: {self.system_stats['total_connections']}")

            # 在 disconnect 之后异步广播更新的系统状态
            if self._monitor_manager:
                asyncio.create_task(self._monitor_manager.broadcast({
                    "type": "system_status_update",
                    "data": self.system_stats,
                    "timestamp": datetime.now().isoformat()
                }))
        else:
            logger.warning(f"尝试断开连接的客户端 {client_id} 已不在活跃连接列表中。")

    async def close_all_connections(self):
        """
        【新增方法】
        安全地关闭所有活跃的 WebSocket 连接。
        """
        logger.info("ConnectionManager: 正在关闭所有活跃连接...")
        # 获取所有客户端ID的列表副本，以便在迭代时安全地修改字典
        client_ids_to_disconnect = list(self.active_connections.keys())
        for client_id in client_ids_to_disconnect:
            # 调用 disconnect 方法，它会处理心跳任务取消和从 active_connections 中移除
            await self.disconnect(client_id)
        logger.info(f"ConnectionManager: 所有 {len(client_ids_to_disconnect)} 个连接已关闭。")

    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """
        向指定的单个客户端发送 JSON 消息。
        """
        connection_info = self.active_connections.get(client_id)
        if connection_info:
            websocket, meeting_id = connection_info
            try:
                if websocket.client_state != WebSocketState.CONNECTED:
                    raise WebSocketDisconnect(f"WebSocket {client_id} 处于非连接状态: {websocket.client_state.name}")
                
                await websocket.send_json(message)
                logger.debug(f"向客户端 {client_id} 发送个人消息 (类型: {message.get('type', '未知类型')})")
            except WebSocketDisconnect:
                logger.warning(f"客户端 {client_id} 已断开连接，无法发送个人消息。正在清理。")
                await self.disconnect(client_id, meeting_id, expected_websocket=websocket) # 使用正确的 meeting_id 和 websocket 进行清理
            except Exception as e:
                logger.error(f"向客户端 {client_id} 发送个人消息失败: {e}", exc_info=True)
                await self.disconnect(client_id, meeting_id) # 即使不是 WebSocketDisconnect，如果发送失败，也视为需要清理
        else:
            logger.warning(f"客户端 {client_id} 不在线，无法发送个人消息。")

    async def send_message_to_meeting(self, message: Dict[str, Any], meeting_id: str):
        """
        向特定会议ID的所有连接广播 JSON 消息。
        """
        # 动态获取会议中的客户端ID列表
        clients_in_meeting = [cid for cid, (_, mid) in self.active_connections.items() if mid == meeting_id]

        if not clients_in_meeting:
            logger.warning(f"会议 {meeting_id} 没有活跃连接，无法广播。")
            return

        disconnected_clients_info = []
        for client_id in clients_in_meeting:
            connection_info = self.active_connections.get(client_id)
            if connection_info:
                websocket, _ = connection_info # meeting_id 已知
                try:
                    if websocket.client_state != WebSocketState.CONNECTED:
                        raise WebSocketDisconnect(f"会议广播目标 WebSocket {client_id} 处于非连接状态: {websocket.client_state.name}")
                    
                    await websocket.send_json(message)
                    logger.debug(f"已向会议 {meeting_id} 中的客户端 {client_id} 发送消息 (类型: {message.get('type')})")
                except WebSocketDisconnect:
                    logger.warning(f"向会议 {meeting_id} 广播时客户端 {client_id} 已断开连接。")
                    disconnected_clients_info.append((client_id, websocket))
                except Exception as e:
                    logger.error(f"向会议 {meeting_id} 的客户端 {client_id} 广播消息失败: {e}", exc_info=True)
                    disconnected_clients_info.append((client_id, websocket))
            else:
                logger.warning(f"会议 {meeting_id} 中的客户端 {client_id} 不在 active_connections 中，可能已断开。")

        # 移除已断开的客户端
        for client_id_to_disconnect, ws_to_disconnect in disconnected_clients_info:
            await self.disconnect(client_id_to_disconnect, meeting_id, expected_websocket=ws_to_disconnect)
            logger.debug(f"从活跃连接中移除 {client_id_to_disconnect} (会议广播时发现断开)。")

    async def broadcast(self, message: Dict[str, Any]):
        """
        向所有活跃的 WebSocket 客户端广播 JSON 消息。
        通常用于系统状态更新等。
        """
        disconnected_clients_info = [] # 存储 (client_id, websocket)
        
        # 迭代副本以安全地删除
        for client_id, (websocket, meeting_id) in list(self.active_connections.items()):
            try:
                if websocket.client_state != WebSocketState.CONNECTED:
                    raise WebSocketDisconnect(f"广播目标 WebSocket {client_id} 处于非连接状态: {websocket.client_state.name}")
                
                await websocket.send_json(message)
            except WebSocketDisconnect:
                logger.warning(f"广播时客户端 {client_id} (会议 {meeting_id}) 已断开连接。")
                disconnected_clients_info.append((client_id, websocket))
            except Exception as e:
                logger.error(f"向客户端 {client_id} (会议 {meeting_id}) 广播消息失败: {e}", exc_info=True)
                disconnected_clients_info.append((client_id, websocket))

        # 移除已断开的客户端
        for client_id, ws in disconnected_clients_info:
            # 这里我们不知道 meeting_id，但 disconnect 方法可以处理
            await self.disconnect(client_id, expected_websocket=ws)
            logger.debug(f"从活跃连接中移除 {client_id} (广播时发现断开)。")

    async def send_ping_periodically(self, client_id: str, websocket: WebSocket, meeting_id: str, interval: int):
        """每隔 'interval' 秒发送一个 ping 帧"""
        try:
            while True: # 循环直到任务被取消或 WebSocket 断开
                # 检查连接是否仍然活跃，且是当前 websocket 实例
                current_connection_info = self.active_connections.get(client_id)
                if not current_connection_info or current_connection_info[0] != websocket:
                    logger.debug(f"客户端 {client_id} 连接已不在活动列表中或已更换，停止心跳。")
                    break # 退出循环

                logger.debug(f"发送 WebSocket ping 到客户端 {client_id} (会议: {meeting_id})")
                try:
                    # 使用 send_bytes 发送 ping 帧
                    await websocket.send_bytes(b'ping')
                except WebSocketDisconnect:
                    logger.info(f"客户端 {client_id} WebSocket 连接已断开，停止发送心跳。")
                    break # 退出循环
                except RuntimeError as e:
                    # 例如 "WebSocket is closed"
                    logger.warning(f"发送 ping 到客户端 {client_id} 失败 (RuntimeError: {e}). 停止心跳。")
                    break # 退出循环
                except Exception as e:
                    logger.error(f"发送心跳失败到客户端 {client_id}: {e}", exc_info=True)
                    break # 退出循环

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info(f"客户端 {client_id} (会议: {meeting_id}) 的心跳任务被取消。")
        except Exception as e:
            logger.error(f"心跳任务未知错误为客户端 {client_id} (会议: {meeting_id}): {e}", exc_info=True)
        finally:
            # 确保在任务结束时清理 ping_tasks
            if client_id in self.ping_tasks and self.ping_tasks[client_id] == asyncio.current_task():
                del self.ping_tasks[client_id]
                logger.debug(f"客户端 {client_id} 的心跳任务已从 ping_tasks 中移除。")

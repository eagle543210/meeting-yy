# M:\meeting\services\neo4j_service.py

import logging
from neo4j import AsyncGraphDatabase, AsyncSession, Query
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)

class Neo4jService:
    """
    负责与 Neo4j 数据库交互的服务类。
    提供连接管理和 Cypher 查询执行功能。
    """
    def __init__(self, uri: str, user: str, password: str):
        logger.info("初始化 Neo4jService...")
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[AsyncGraphDatabase.driver] = None
        self._connected: bool = False

    async def connect(self):
        """
        异步连接到 Neo4j 数据库。
        """
        if self._connected and self.driver:
            logger.info("Neo4j 已连接，跳过重复连接。")
            return

        logger.info(f"尝试连接到 Neo4j 数据库: {self.uri}...")
        try:
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 尝试验证连接
            await self.driver.verify_connectivity()
            self._connected = True
            logger.info("🎉 Neo4j 数据库连接成功。")
        except Exception as e:
            self._connected = False
            logger.critical(f"❌ Neo4j 数据库连接失败: {e}", exc_info=True)
            raise RuntimeError(f"无法连接到 Neo4j 数据库: {e}") from e

    async def close(self):
        """
        异步关闭 Neo4j 数据库连接。
        """
        if self.driver:
            logger.info("正在关闭 Neo4j 数据库连接...")
            await self.driver.close()
            self.driver = None
            self._connected = False
            logger.info("Neo4j 数据库连接已关闭。")

    def is_connected(self) -> bool:
        """
        检查 Neo4j 数据库是否已连接。
        """
        return self._connected

    async def run_query(self, query: str, parameters: Optional[Dict] = None, write: bool = False) -> List[Dict[str, Any]]:
        """
        异步执行 Cypher 查询。
        Args:
            query (str): 要执行的 Cypher 查询字符串。
            parameters (Optional[Dict]): 查询参数。
            write (bool): 如果为 True，则以写模式执行事务；否则以读模式执行。
        Returns:
            List[Dict[str, Any]]: 查询结果列表，每个元素是一个字典。
        Raises:
            RuntimeError: 如果数据库未连接或查询执行失败。
        """
        if not self.is_connected() or not self.driver:
            raise RuntimeError("Neo4j 数据库未连接。")

        # logger.debug(f"执行 Cypher 查询 (写入模式: {write}): {query}...")
        # logger.debug(f"传入的参数类型: {type(parameters)}, 值: {parameters}") 

        try:
            async with self.driver.session() as session:
                # 定义一个异步事务函数
                async def execute_transaction(tx, q, p):
                    # 关键修正：明确使用 parameters 关键字参数
                    # 再次强调：确保 p 是字典或 None
                    logger.debug(f"事务内 tx.run() 接收的参数类型: {type(p)}, 值: {p}") 
                    result_cursor = await tx.run(q, parameters=p) 
                    return await result_cursor.data()

                if write:
                    result = await session.execute_write(execute_transaction, query, parameters)
                else:
                    result = await session.execute_read(execute_transaction, query, parameters)
            logger.debug("Cypher 查询执行完成。")
            return result
        except Exception as e:
            logger.error(f"执行 Cypher 查询失败: {e}", exc_info=True)
            raise RuntimeError(f"Neo4j 查询失败: {e}") from e

    async def initialize_schema(self):
        """
        初始化 Neo4j 数据库的 Schema，包括创建唯一约束和索引。
        """
        logger.info("正在初始化 Neo4j Schema...")
        constraints_and_indexes = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Meeting) REQUIRE m.id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Relationship) ON (r.type)"
        ]
        for cql in constraints_and_indexes:
            try:
                await self.run_query(cql, write=True)
                logger.info(f"Schema 初始化成功: {cql}")
            except Exception as e:
                logger.error(f"Schema 初始化失败: {cql} - {e}", exc_info=True)
                # 即使失败，也尝试继续其他约束，但记录错误
        logger.info("Neo4j Schema 初始化完成。")

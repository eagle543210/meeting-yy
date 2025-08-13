# services/neo4j_connector.py
from neo4j import GraphDatabase
from config.settings import settings
from typing import List, Dict, Any

class Neo4jConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASS)
        )
    
    def execute_query(self, query: str, params: Dict = None) -> List[Dict[str, Any]]:
        """执行Cypher查询"""
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def close(self):
        """关闭连接"""
        self.driver.close()

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            self.execute_query("RETURN 1 AS test")
            return True
        except Exception:
            return False

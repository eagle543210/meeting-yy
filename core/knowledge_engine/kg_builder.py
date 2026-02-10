# M:\meeting\core\knowledge_engine\kg_builder.py

import logging
from typing import List, Tuple, Dict, Any, Optional
import asyncio

from services.neo4j_service import Neo4jService # 导入 Neo4jService
from services.llm_service import LLMModel # 导入 LLMModel
from config.settings import settings # 导入 settings

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """
    负责构建和查询知识图谱的类。
    它使用 Neo4jService 与 Neo4j 数据库交互。
    """
    def __init__(self, neo4j_service: Neo4jService, llm_model: LLMModel, settings_obj: settings): # <-- 修正：添加 llm_model 和 settings_obj
        logger.info("初始化 KnowledgeGraphBuilder...")
        self.neo4j_service = neo4j_service
        self.llm_model = llm_model # 存储 LLM 模型实例
        self.settings = settings_obj # 存储 settings 对象
        logger.info("KnowledgeGraphBuilder 初始化完成。")

    async def update_graph(self, triples: List[Tuple[str, str, str]]):
        """
        将一组三元组 (实体1, 关系, 实体2) 添加或更新到知识图谱中。
        Args:
            triples (List[Tuple[str, str, str]]): 要添加的三元组列表。
        """
        if not self.neo4j_service.is_connected():
            logger.error("Neo4j 服务未连接，无法更新知识图谱。")
            raise RuntimeError("知识图谱服务未准备好。")

        logger.info(f"正在更新知识图谱，添加 {len(triples)} 个三元组...")
        query = """
        UNWIND $triples AS triple
        MERGE (e1:Entity {name: triple[0]})
        MERGE (e2:Entity {name: triple[2]})
        MERGE (e1)-[r:Relationship {type: triple[1]}]->(e2)
        RETURN e1.name, type(r), e2.name
        """
        parameters = {"triples": triples}
        try:
            results = await self.neo4j_service.run_query(query, parameters, write=True)
            logger.info(f"知识图谱更新成功。已处理 {len(results)} 个关系。")
        except Exception as e:
            logger.error(f"更新知识图谱失败: {e}", exc_info=True)
            raise RuntimeError(f"更新知识图谱失败: {e}") from e

    async def is_database_empty(self) -> bool:
        """
        检查知识图谱数据库是否为空。
        Returns:
            bool: 如果数据库中没有实体，则为 True；否则为 False。
        """
        logger.info("KnowledgeGraphBuilder: 正在执行异步 is_database_empty 检查...")
        if not self.neo4j_service.is_connected():
            logger.error("Neo4j 服务未连接，无法检查数据库是否为空。")
            return True # 视为为空，因为无法访问

        query = "MATCH (n:Entity) RETURN count(n) AS count"
        try:
            results = await self.neo4j_service.run_query(query, write=False)
            count = results[0]["count"] if results else 0
            is_empty = count == 0
            logger.info(f"知识图谱数据库是否为空: {is_empty} (实体数量: {count})")
            return is_empty
        except Exception as e:
            logger.error(f"检查知识图谱数据库是否为空失败: {e}", exc_info=True)
            return True # 出现错误时，保守地认为数据库为空

    async def query_entity(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        """
        查询知识图谱中与给定实体相关的实体和关系。
        Args:
            entity_name (str): 要查询的实体名称。
            depth (int): 查询的深度（默认为1）。
        Returns:
            Dict[str, Any]: 包含查询结果的字典。
        """
        if not self.neo4j_service.is_connected():
            logger.error("Neo4j 服务未连接，无法查询知识图谱。")
            raise RuntimeError("知识图谱服务未准备好。")

        logger.info(f"查询实体 '{entity_name}'，深度 {depth}...")
        query = f"""
        MATCH (start:Entity)-[r*1..{depth}]-(end:Entity)
        WHERE start.name = $entityName
        RETURN start.name AS startNode, type(r) AS relationshipType, end.name AS endNode
        """
        parameters = {"entityName": entity_name}
        try:
            results = await self.neo4j_service.run_query(query, parameters, write=False)
            
            nodes = set()
            relationships = []
            
            for record in results:
                nodes.add(record["startNode"])
                nodes.add(record["endNode"])
                relationships.append({
                    "start": record["startNode"],
                    "type": record["relationshipType"],
                    "end": record["endNode"]
                })
            
            # 如果没有找到关系，但实体本身存在，也应该返回该实体
            if not relationships:
                check_entity_query = "MATCH (e:Entity) WHERE e.name = $entityName RETURN e.name"
                entity_exists = await self.neo4j_service.run_query(check_entity_query, parameters, write=False)
                if entity_exists:
                    nodes.add(entity_name)

            logger.info(f"实体 '{entity_name}' 查询完成。找到 {len(nodes)} 个节点和 {len(relationships)} 个关系。")
            return {
                "entity": entity_name,
                "nodes": list(nodes),
                "relationships": relationships
            }
        except Exception as e:
            logger.error(f"查询实体 '{entity_name}' 失败: {e}", exc_info=True)
            raise ValueError(f"知识图谱查询失败: {e}") from e

    async def extract_and_store_entities(self, text: str) -> List[str]:
        """
        使用 LLM 从文本中提取实体，并将其存储到知识图谱中。
        Args:
            text (str): 要提取实体的文本。
        Returns:
            List[str]: 提取到的实体列表。
        """
        if not self.llm_model.is_model_loaded():
            logger.error("LLM 模型未加载，无法提取实体。")
            return []

        prompt = f"从以下文本中提取所有主要实体（例如，人名、地名、组织、关键概念、产品名称等），以逗号分隔的列表形式返回。如果没有，则回答'无实体'。\n\n文本: {text}\n\n实体:"
        try:
            response = await self.llm_model.generate_text(
                prompt,
                max_new_tokens=150, # 适当的长度限制
                temperature=0.5 # 较低的温度以获得更确定的结果
            )
            
            # 解析 LLM 的响应，提取实体
            entities_raw = [e.strip() for e in response.split(',') if e.strip() and e.strip().lower() != '无实体']
            
            # 简单去重
            extracted_entities = list(set(entities_raw))

            # 暂时不在这里直接构建三元组，只返回实体。
            # 如果需要在这里直接构建三元组并存储，则需要 LLM 能够生成结构化的三元组。
            # 目前 SmartMeetingAssistant 会调用此方法获取实体，然后自行决定如何使用。
            
            logger.info(f"从文本中提取的实体: {extracted_entities}")
            return extracted_entities
        except Exception as e:
            logger.error(f"提取实体失败: {e}", exc_info=True)
            return []


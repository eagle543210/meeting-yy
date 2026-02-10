# core/knowledge_engine/__init__.py
from .kg_builder import KnowledgeGraphBuilder
from typing import Optional, Dict, List

__all__ = ['KnowledgeGraphBuilder', 'KnowledgeProcessor']

class KnowledgeProcessor:
    def __init__(self):
        """初始化知识处理器"""
        from .kg_builder import KnowledgeGraphBuilder  # 相对导入
        self.kg_builder = KnowledgeGraphBuilder()
    
    def extract_triples(self, text: str) -> List[tuple]:
        """从文本中提取三元组"""
        # 实现您的提取逻辑
        return []
    
    def query(self, entity: str, depth: int = 1) -> dict:
        """
        查询知识图谱的统一接口
        
        参数:
            entity: 要查询的实体名称
            depth: 查询深度(默认1)
        """
        try:
            return self.kg_builder.query_entity(entity, depth)
        except Exception as e:
            logging.error(f"Query failed for entity {entity}: {str(e)}")
            return {
                "entity": entity,
                "related_entities": [],
                "error": str(e)
            }
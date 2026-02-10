# M:\meeting\backend\BackendCoordinator.py

# Python 标准库导入
import asyncio # 用于异步操作和信号量
import logging # 用于日志记录
import uuid # 用于生成唯一标识符
from datetime import datetime # 用于日期和时间处理
import re # 用于正则表达式操作
from typing import Dict, Any, Optional, List, Tuple, AsyncIterator # 导入 AsyncIterator
from bson.objectid import ObjectId # 导入 ObjectId
import json # 导入 json 以用于 Milvus expr
from pymilvus import SearchResult, Hit
# 第三方库导入
import numpy as np # 用于数值计算，特别是音频数据处理

# 项目内部模块导入
from config.settings import settings # 项目配置设置
from services.mongodb_manager import MongoDBManager # MongoDB 数据库服务
from services.milvus_service import MilvusManager # Milvus 向量数据库服务（现在是通用型）
from services.llm_service import LLMModel # 大语言模型服务
from services.embedding_service import BGEEmbeddingModel # 嵌入模型服务
from services.neo4j_service import Neo4jService # Neo4j 图数据库服务
from services.permission_service import PermissionService # 权限管理服务
from core.speech_to_text.stt_processor import SpeechToTextProcessor # 语音转文字处理服务
from services.voiceprint_service import VoiceprintService # 声纹识别服务
from services.summary_service import SummaryService # 总结服务
from core.data_processing.minute_generator import MinuteGenerator # 智能会议助手
from backend.connection_manager import ConnectionManager # WebSocket 连接管理器
from backend.monitor_manager import MonitorManager # 监控管理器
from models import User, UserRole, Permission # 数据模型定义 (User, UserRole, Permission)
from core.knowledge_engine.kg_builder import KnowledgeGraphBuilder # 知识图谱构建器

logger = logging.getLogger(__name__)

class BackendCoordinator:
    """
    BackendCoordinator 协调所有后端服务，包括数据库、AI 模型和实时通信。
    它充当一个外观模式（Facade），简化了前端和复杂后端逻辑之间的交互。
    """
    def __init__(self,
                 settings_obj: settings,
                 mongodb_manager: MongoDBManager,
                 voice_milvus_manager: MilvusManager, 
                 meeting_milvus_manager: MilvusManager, 
                 llm_model: LLMModel,
                 bge_model: BGEEmbeddingModel,
                 neo4j_service: Neo4jService,
                 permission_service: PermissionService,
                 stt_processor: SpeechToTextProcessor,
                 voiceprint_service: VoiceprintService,
                 summary_service: SummaryService,
                 connection_manager: ConnectionManager):
        
        logger.info("初始化后端协调器...")
        self.settings = settings_obj
        self.mongodb_manager = mongodb_manager
        self.voice_milvus_manager = voice_milvus_manager 
        self.meeting_milvus_manager = meeting_milvus_manager 
        self.llm_model = llm_model
        self.bge_model = bge_model
        self.neo4j_service = neo4j_service
        self.permission_service = permission_service
        self.stt_processor = stt_processor
        self.voiceprint_service = voiceprint_service
        self.summary_service = summary_service 

        self.connection_manager = connection_manager
        logger.info("ConnectionManager 已在 BackendCoordinator 中设置。")

        # 实例化 MinuteGenerator，并传入 ConnectionManager
        self.meeting_assistant = MinuteGenerator(
            settings_obj=self.settings,
            mongodb_manager=self.mongodb_manager,
            meeting_milvus_manager=self.meeting_milvus_manager, 
            llm_model=self.llm_model,
            bge_model=self.bge_model,
            neo4j_service=self.neo4j_service,
            stt_processor=self.stt_processor,
            voiceprint_service=self.voiceprint_service,
            summary_service=self.summary_service,
            connection_manager=self.connection_manager
        )
        logger.info("MinuteGenerator 实例已创建。")

        # 实例化 KnowledgeGraphBuilder (此处需要 LLM 和 Neo4jService)
        self.kg_builder = KnowledgeGraphBuilder(
            neo4j_service=self.neo4j_service, 
            llm_model=self.llm_model,
            settings_obj=self.settings
        )
        logger.info("知识图谱构建器实例已创建。")

        self.monitor_manager: Optional[MonitorManager] = None 

        # LLM 并发控制信号量，使用配置值
        self._llm_semaphore = asyncio.Semaphore(self.settings.LLM_CONCURRENCY_LIMIT) 
        logger.info(f"LLM 并发信号量已初始化为: {self._llm_semaphore._value}")

        logger.info("后端协调器初始化完成。")

    async def close_services(self):
        """
        异步关闭所有连接的服务。
        在应用程序关闭时调用，以确保所有资源被正确释放。
        """
        logger.info("正在关闭所有后端服务...")
        
        # 关闭 MongoDB 连接
        if self.mongodb_manager:
            await self.mongodb_manager.close()
            logger.info("MongoDB 连接已关闭。")
        
        # 关闭 Milvus 连接
        if self.voice_milvus_manager:
            await self.voice_milvus_manager.close()
            logger.info("声纹 Milvus 连接已关闭。")
        
        if self.meeting_milvus_manager:
            await self.meeting_milvus_manager.close()
            logger.info("会议 Milvus 连接已关闭。")
            
        # 关闭 Neo4j 连接
        if self.neo4j_service:
            await self.neo4j_service.close()
            logger.info("Neo4j 连接已关闭。")
        
        # 关闭 LLM 模型
        if self.llm_model:
            await self.llm_model.close()
            logger.info("LLM 模型已卸载。")

        logger.info("所有后端服务已成功关闭。")

    def set_monitor_manager(self, manager: MonitorManager):
        """设置 MonitorManager 实例。"""
        self.monitor_manager = manager
        logger.info("MonitorManager 已在 BackendCoordinator 中设置。")

    async def register_voice(self, audio_data: np.ndarray, sample_rate: int, user_name: str, role_str: str) -> Dict[str, Any]:
        """
        注册用户的声纹和信息。
        Args:
            audio_data (np.ndarray): 音频数据。
            sample_rate (int): 音频采样率。
            user_name (str): 用户名。
            role_str (str): 用户角色字符串。
        Returns:
            Dict[str, Any]: 注册结果。
        Raises:
            RuntimeError: 如果所需服务未初始化。
            ValueError: 如果声纹嵌入无法生成或出现声纹冲突。
        """
        logger.info(f"后端协调器: 正在尝试为用户 '{user_name}' (角色: {role_str}) 注册声纹...")
        if not self.voiceprint_service or not self.mongodb_manager:
            raise RuntimeError("声纹注册所需服务(VoiceprintService/MongoDBManager)未完全初始化。")

        try:
            # 生成一个唯一的 user_id 作为主键
            user_id = str(uuid.uuid4()) 
            
            # 调用 VoiceprintService 注册声纹。VoiceprintService 在内部处理 Milvus 和 MongoDB。
            registration_result = await self.voiceprint_service.register_voice(
                audio_data, sample_rate, user_id, user_name, role_str
            )

            registered_user_id = registration_result.get("user_id")
            
            return {
                "status": "registered",
                "message": "声纹注册成功",
                "user_id": registered_user_id,
                "user_name": user_name,
                "role": role_str
            }
        # 优化建议:
        # 3. 这里的异常捕获使用了通用的 `Exception`。虽然它可以捕获所有错误，但更具体的异常捕获（如 `ValueError`、`MilvusException`）可以提供更精确的错误处理和日志记录。
        except Exception as e:
            logger.error(f"后端协调器: 用户 '{user_name}' 的声纹注册失败: {e}", exc_info=True)
            raise RuntimeError(f"声纹注册失败: {e}") from e

    async def process_meeting(self, audio_data: np.ndarray, sample_rate: int, transcript: str) -> Dict[str, Any]:
        """
        处理完整的会议数据（离线模式），以生成总结、行动项等。
        """
        if not self.meeting_assistant:
            logger.error("MinuteGenerator (会议处理服务) 未初始化。")
            raise RuntimeError("会议处理所需服务未完全初始化。")

        logger.info("开始处理完整的会议数据...")
        try:
            result = await self.meeting_assistant.process_meeting(audio_data, sample_rate, transcript)
            logger.info("完整会议数据处理完成。")
            return result
        # 优化建议:
        # 4. 同样，这里使用了通用的 `Exception`。如果能根据 `meeting_assistant` 可能抛出的具体异常进行捕获，会更健壮。
        except Exception as e:
            logger.error(f"处理完整的会议数据失败: {e}", exc_info=True)
            raise RuntimeError(f"处理完整的会议数据失败: {e}") from e

    async def query_knowledge_graph(self, entity_name: str, depth: int = 1) -> List[Dict[str, Any]]: 
        """
        查询知识图谱，以获取与给定实体相关的信息。
        Args:
            entity_name (str): 要查询的实体名称。
            depth (int): 查询深度。
        Returns:
            List[Dict[str, Any]]: 知识图谱查询结果。
        """
        if not self.neo4j_service: 
            logger.error("Neo4jService 未初始化。")
            raise RuntimeError("知识图谱服务未初始化。")

        logger.info(f"正在查询知识图谱实体: '{entity_name}', 深度: {depth}")
        try:
            cypher_query = f"""
            MATCH (e:Entity {{name: $entity_name}})-[r*1..{depth}]-(related)
            RETURN e, r, related
            """
            parameters = {"entity_name": entity_name} 
            
            raw_results = await self.neo4j_service.run_query(cypher_query, parameters, write=False) 
            
            logger.debug(f"从 Neo4j 获得原始结果 ({len(raw_results)} 条记录): {raw_results}")

            formatted_results = []
            seen_identifiers = set() 

            for i, record in enumerate(raw_results):
                logger.debug(f"正在处理原始结果记录 {i}: {record}")

                if 'e' in record and isinstance(record['e'], dict):
                    node_name = record['e'].get('name')
                    if node_name and node_name not in seen_identifiers: 
                        formatted_results.append({
                            "type": "node",
                            "labels": ["Entity"], 
                            "properties": {"name": node_name}
                        })
                        seen_identifiers.add(node_name)
                        logger.debug(f"已添加节点 'e': {node_name}")
                else:
                    logger.debug(f"记录 {i}: 'e' 不存在或不是字典: {record.get('e')}")
                
                if 'r' in record and record['r'] is not None:
                    relationships_to_process = record['r'] if isinstance(record['r'], list) else [record['r']]
                    for rel_data in relationships_to_process:
                        if isinstance(rel_data, tuple) and len(rel_data) == 3:
                            start_node_dict = rel_data[0]
                            rel_type = rel_data[1]
                            end_node_dict = rel_data[2]
                            
                            start_node_name = start_node_dict.get('name')
                            end_node_name = end_node_dict.get('name')

                            rel_identifier = f"{start_node_name}-{rel_type}-{end_node_name}"
                            
                            if rel_identifier not in seen_identifiers:
                                formatted_results.append({
                                    "type": "relationship",
                                    "start_node_name": start_node_name, 
                                    "end_node_name": end_node_name, 
                                    "relationship_type": rel_type,
                                    "properties": {} 
                                })
                                seen_identifiers.add(rel_identifier)
                                logger.debug(f"已添加关系 'r': {rel_identifier}")
                        else:
                            logger.debug(f"记录 {i}: 'r' 中的元素不是预期的元组类型: {rel_data}")
                else:
                    logger.debug(f"记录 {i}: 'r' 不存在或为 None: {record.get('r')}")

                if 'related' in record and isinstance(record['related'], dict):
                    node_name = record['related'].get('name')
                    if node_name and node_name not in seen_identifiers:
                        formatted_results.append({
                            "type": "node",
                            "labels": ["Entity"], 
                            "properties": {"name": node_name}
                        })
                        seen_identifiers.add(node_name)
                        logger.debug(f"已添加节点 'related': {node_name}")
                else:
                    logger.debug(f"记录 {i}: 'related' 不存在或不是字典: {record.get('related')}")
            
            logger.info(f"知识图谱查询完成，返回 {len(formatted_results)} 条格式化结果。")
            return formatted_results
        # 优化建议:
        # 5. 这里的异常捕获使用了通用的 `Exception`。对于数据库查询，捕获特定的 Neo4j 驱动异常（如 `neo4j.exceptions.ClientError`）会更具针对性。
        except Exception as e:
            logger.error(f"知识图谱查询失败: {e}", exc_info=True)
            raise ValueError(f"知识图谱查询失败: {e}") from e

    async def update_user_role_from_ws(self, user_id: str, new_role_str: str) -> Dict[str, Any]:
        """
        从 WebSocket 请求更新用户角色。
        """
        if not self.mongodb_manager:
            logger.error("MongoDBManager 未初始化。")
            return {"status": "error", "message": "数据库服务未准备就绪。"}

        logger.info(f"后端协调器: 正在尝试通过 WS 将用户 '{user_id}' 的角色更新为 '{new_role_str}'...")
        try:
            new_role_enum = UserRole(new_role_str.upper())
            await self.mongodb_manager.update_user_role(user_id, new_role_enum)
            
            logger.info(f"用户 '{user_id}' 的角色已通过 WS 请求成功更新为 '{new_role_str}'。")
            return {
                "type": "user_role_updated",
                "status": "success",
                "message": f"用户 {user_id} 角色更新成功",
                "userId": user_id,
                "newRole": new_role_str,
                "timestamp": datetime.now().isoformat()
            }
        # 优化建议:
        # 6. 这里捕获了通用异常。如果 `UserRole` 转换失败，会抛出 `ValueError`；如果数据库操作失败，可能会抛出不同的异常。针对性地处理这些异常会更好。
        except Exception as e:
            logger.error(f"WS: 更新用户 '{user_id}' 角色时出错: {e}", exc_info=True)
            return {
                "type": "error",
                "status": "failure",
                "message": f"更新用户角色时发生内部错误: {e}",
                "user_id": user_id,
                "new_role": new_role_str,
                "timestamp": datetime.now().isoformat()
            }

    async def _extract_kg_from_transcript(self, transcript_text: str) -> List[Tuple[str, str, str]]:
        """
        使用 LLM 从会议转录文本中提取知识图谱三元组 (主语, 谓语, 宾语)。
        此方法用于后台知识图谱构建，因此是非流式的。
        """
        logger.info("后端协调器: 正在使用 LLM 从转录文本中提取知识图谱三元组...")
        prompt = f"""
        请从以下会议转录文本中提取关键实体及其关系，并将它们列为三元组（主语、谓语、宾语）。
        每个三元组应占一行，例如：
        (张三, 负责, 项目A)
        (会议, 讨论了, 预算)
        (项目A, 截止日期, 2025-12-31)
        
        请只返回三元组列表，不要包含任何额外的解释或描述。

        会议转录文本:
        {transcript_text}
        """
        try:
            # 使用信号量控制 LLM 调用并发
            async with self._llm_semaphore:
                # 调用 LLMModel.generate_text 的非流式版本
                # 优化建议:
                # 7. 这里使用了 `async for chunk in ... stream=False` 的模式，但 `stream=False` 意味着响应是单次返回的。
                #    直接使用 `await self.llm_model.generate_text(prompt, stream=False)` 并处理其返回值（例如一个字符串）可能更直接。
                full_response_chunks = []
                async for chunk in self.llm_model.generate_text(prompt, stream=False): 
                    full_response_chunks.append(chunk)
                kg_triples_raw = "".join(full_response_chunks)
            
            triples = []
            pattern = re.compile(r'\((.*?),\s*(.*?),\s*(.*?)\)')
            
            for line in kg_triples_raw.split('\n'):
                line = line.strip()
                match = pattern.match(line)
                if match:
                    subject = match.group(1).strip()
                    predicate = match.group(2).strip()
                    obj = match.group(3).strip()
                    triples.append((subject, predicate, obj))
                else:
                    if line:
                        logger.warning(f"LLM 生成的三元组格式不正确，跳过: {line}")
            logger.info(f"后端协调器: 已成功从转录文本中提取 {len(triples)} 个知识图谱三元组。")
            return triples
        # 优化建议:
        # 8. 这里的 `Exception` 捕获也应考虑更具体的异常类型，例如 LLM 调用失败可能抛出的网络或API错误。
        except Exception as e:
            logger.error(f"后端协调器: 从转录文本中提取知识图谱三元组失败: {e}", exc_info=True)
            return []

    async def process_meeting_for_knowledge_graph(self, meeting_id: str) -> Dict[str, Any]:
        """
        检索指定会议的转录文本，提取知识图谱信息，并将其存储在 Neo4j 中。
        """
        logger.info(f"后端协调器: 正在为会议 '{meeting_id}' 处理知识图谱集成...")
        try:
            full_transcript_entries = await self.mongodb_manager.get_all_transcripts_for_meeting(meeting_id)
            
            full_transcript_text = ""
            for entry in full_transcript_entries:
                speaker = entry.speaker_id
                text = entry.text
                if isinstance(entry.timestamp, str):
                    try:
                        timestamp_dt = datetime.fromisoformat(entry.timestamp)
                    except ValueError:
                        timestamp_dt = None
                else:
                    timestamp_dt = entry.timestamp
                timestamp_str = timestamp_dt.strftime('%H:%M:%S') if timestamp_dt else "未知时间"
                full_transcript_text += f"[{timestamp_str}] {speaker}: {text}\n"

            if not full_transcript_text.strip():
                logger.warning(f"会议 '{meeting_id}' 没有转录内容，无法生成知识图谱。")
                return {"status": "warning", "message": "会议没有转录内容，无法生成知识图谱。"}

            kg_triples = await self._extract_kg_from_transcript(full_transcript_text)
            
            if not kg_triples:
                logger.warning(f"无法从会议 '{meeting_id}' 的转录文本中提取任何知识图谱三元组。")
                return {"status": "warning", "message": "无法从转录文本中提取任何知识图谱三元组。"}

            await self.kg_builder.update_graph(kg_triples)
            logger.info(f"后端协调器: 已成功将会议 '{meeting_id}' 的 {len(kg_triples)} 个知识图谱三元组添加到 Neo4j。")

            return {"status": "success", "message": f"会议 '{meeting_id}' 已成功链接到知识图谱，添加了 {len(kg_triples)} 个三元组。"}
        # 优化建议:
        # 9. 这里也使用了通用的 `Exception`，更具体的异常捕获能帮助您区分是数据库错误、LLM错误还是其他逻辑错误。
        except Exception as e:
            logger.error(f"后端协调器: 为会议 '{meeting_id}' 处理知识图谱集成失败: {e}", exc_info=True)
            raise RuntimeError(f"知识图谱集成失败: {e}")

    # --- 辅助方法: 从查询中提取人物和话题 ---
    def _extract_entities_from_query(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取预定义的人物和话题。
        """
        persons = []
        topics = []
        
        # 从配置中读取预定义的人物和话题
        all_persons = self.settings.PREDEFINED_PERSONS
        all_topics = self.settings.PREDEFINED_TOPICS

        for p in all_persons:
            if p in query:
                persons.append(p)
        for t in all_topics:
            if t in query:
                topics.append(t)
                
        return {"persons": persons, "topics": topics}

     # --- 辅助方法: 通用 Milvus 查询函数 ---
    async def _query_milvus_and_mongo(
        self, 
        query_text: str, 
        top_k: int = 5,
        mongo_filter: Optional[Dict[str, Any]] = None 
    ) -> List[Dict[str, Any]]:
        """
        根据用户查询，查询 Milvus（会议文本嵌入集合）以获取相关文本，
        并从 MongoDB 中检索原始数据。
        可以传递 mongo_filter 来限制搜索范围。
        """
        if not self.meeting_milvus_manager or not self.meeting_milvus_manager.is_connected or self.mongodb_manager.db is None or self.bge_model is None:
            logger.error("Milvus (会议文本嵌入)/MongoDB/BGE 服务未完全初始化，无法执行查询。")
            return []

        logger.info(f"--- 正在执行 Milvus 向量查询 (集合: {self.settings.MILVUS_MEETING_COLLECTION_NAME}): '{query_text}' ---")
        try:
            # 调用 BGE 模型生成查询嵌入
            query_embedding_list = await self.bge_model.get_embedding(query_text) 
            
            if query_embedding_list is None:
                logger.warning("BGE 模型未生成任何嵌入，或生成失败。")
                return []

            milvus_expr = ""
            if mongo_filter and "_id" in mongo_filter and "$in" in mongo_filter["_id"]:
                # Milvus 筛选表达式需要字符串形式的ID
                mongo_ids_list = [str(obj_id) for obj_id in mongo_filter["_id"]["$in"]]
                milvus_expr = f"mongo_id in {json.dumps(mongo_ids_list)}" if mongo_ids_list else ""
                logger.info(f"Milvus 筛选表达式: {milvus_expr}")

            # 使用 meeting_milvus_manager 进行搜索，返回 SearchResult 对象
            search_results: SearchResult = await self.meeting_milvus_manager.search_data(
                query_vectors=[query_embedding_list], 
                top_k=top_k,
                expr=milvus_expr,
                output_fields=["mongo_id"] 
            )
            
            retrieved_mongo_ids = []
            
            if not isinstance(search_results, SearchResult):
                logger.error(f"严重错误：milvus_service.search_data() 返回了非预期的对象类型：{type(search_results)}。预计是 pymilvus.SearchResult。")
                return []

            # SearchResult 是一个包含 Hits 列表的列表，通常只有一个
            for hits_list in search_results:
                for hit in hits_list:
                    if not isinstance(hit, Hit):
                        logger.error(f"严重错误：Milvus 搜索结果中的对象类型不正确：{type(hit)}。预计是 pymilvus.Hit。")
                        continue
                    
                    # 访问 hit.entity 来获取 scalar fields
                    mongo_id = hit.entity.get('mongo_id')
                    if mongo_id:
                        retrieved_mongo_ids.append(str(mongo_id))
                    else:
                        logger.warning(f"警告: 搜索结果命中对象 {hit} 的 'entity' 中没有找到 'mongo_id' 字段。")
            
            logger.info(f"Milvus 搜索找到 {len(retrieved_mongo_ids)} 个 mongo_id。")

            if not retrieved_mongo_ids:
                logger.info("从 Milvus 中未检索到任何相关结果。")
                return []

            # 核心修复: 直接使用字符串ID列表进行MongoDB查询，不需要再转换为ObjectId
            cursor = self.mongodb_manager.db[self.settings.MONGO_TRANSCRIPT_COLLECTION_NAME].find(
                {"_id": {"$in": retrieved_mongo_ids}} # <--- 注意这里直接使用了 retrieved_mongo_ids
            )
            mongo_query_results = await cursor.to_list(length=None)

            # 重新排序 MongoDB 文档以匹配 Milvus 的相关性顺序
            mongo_doc_map = {doc['_id']: doc for doc in mongo_query_results}
            final_results = []
            for mid in retrieved_mongo_ids:
                if mid in mongo_doc_map:
                    final_results.append(mongo_doc_map[mid])
                else:
                    logger.warning(f"警告: 在 MongoDB 中未找到 ID 为 {mid} 的文档。")

            return final_results

        except Exception as e:
            logger.error(f"Milvus/MongoDB 查询失败: {e}", exc_info=True)
            return []

    #  混合查询函数 (结合 Neo4j 和 Milvus/MongoDB) ---
    async def _hybrid_query(
        self, 
        query_text: str, 
        top_k_milvus: int = 5
    ) -> List[Dict[str, Any]]:
        """
        [修复] 执行混合查询:
        1. 尝试从查询中提取人物和话题。
        2. 如果提取到实体，则首先查询 Neo4j 以获取相关上下文（例如，会议 ID 或与发言人相关的记录）。
        3. 使用 Neo4j 结果作为过滤器来执行 Milvus 向量搜索。
        4. 如果未提取到实体，则执行直接的 Milvus 向量搜索。
        5. 从 MongoDB 中检索最终文本。
        """
        logger.info(f"--- 正在开始混合查询: '{query_text}' ---")
        extracted_entities = self._extract_entities_from_query(query_text)
        persons = extracted_entities.get("persons", [])
        topics = extracted_entities.get("topics", [])
        
        mongo_filter = None
        
        # 检查 Neo4j 是否可用以及是否有实体需要查询
        if self.neo4j_service.driver and (persons or topics):
            logger.info("在查询中识别出人物或话题，正在尝试从 Neo4j 获取上下文...")
            try:
                # 使用 Neo4j 异步会话
                async with self.neo4j_service.driver.session() as session:
                    cypher_query_parts = []
                    params = {}

                    if persons:
                        cypher_query_parts.append("""
                            MATCH (p:Person)-[:SPOKE_IN|MENTIONED]->(n)
                            WHERE p.name IN $persons_list
                            RETURN
                                CASE WHEN n:Meeting THEN n.id END AS meeting_id,
                                CASE WHEN n:Person THEN n.name END AS speaker_name,
                                CASE WHEN n:Topic THEN n.name END AS topic_name
                        """)
                        params["persons_list"] = persons

                    if topics:
                        cypher_query_parts.append("""
                            MATCH (n)-[:DISCUSSED]->(t:Topic)
                            WHERE t.name IN $topics_list
                            RETURN
                                CASE WHEN n:Meeting THEN n.id END AS meeting_id,
                                CASE WHEN n:Person THEN n.name END AS speaker_name,
                                CASE WHEN n:Topic THEN n.name END AS topic_name
                        """)
                        params["topics_list"] = topics
                    
                    # 组合查询字符串
                    cypher_query = " UNION ".join(cypher_query_parts)
                    
                    if cypher_query:
                        logger.info(f"正在执行 Cypher 查询: {cypher_query}")
                        # 修复：直接使用 'await' 调用异步会话的 run 方法
                        neo4j_results = await session.run(cypher_query, **params)
                        
                        relevant_meeting_ids = set()
                        relevant_speakers = set()
                        
                        # 获取 MongoDB 中已有的会议和发言人ID，用于过滤
                        distinct_meeting_ids_in_mongo = await self.mongodb_manager.get_distinct_meeting_ids()
                        distinct_speakers_in_mongo = await self.mongodb_manager.get_distinct_speakers()

                        # 修复：await session.run() 返回的结果对象可以直接迭代
                        for record in neo4j_results:
                            meeting_id = record.get('meeting_id')
                            speaker_name = record.get('speaker_name')
                            
                            if meeting_id and meeting_id in distinct_meeting_ids_in_mongo:
                                relevant_meeting_ids.add(meeting_id)
                            
                            if speaker_name and speaker_name in distinct_speakers_in_mongo:
                                relevant_speakers.add(speaker_name)
                        
                        logger.info(f"Neo4j 找到相关会议 ID: {relevant_meeting_ids}")
                        logger.info(f"Neo4j 找到相关发言人: {relevant_speakers}")

                        mongo_id_candidates = []
                        if relevant_meeting_ids or relevant_speakers:
                            if relevant_meeting_ids:
                                cursor_meetings = self.mongodb_manager.db[self.settings.MONGO_TRANSCRIPT_COLLECTION_NAME].find(
                                    {"meeting_id": {"$in": list(relevant_meeting_ids)}}, {"_id": 1}
                                )
                                docs_meetings = await cursor_meetings.to_list(length=None)
                                mongo_id_candidates.extend([doc["_id"] for doc in docs_meetings])

                            if relevant_speakers:
                                cursor_speakers = self.mongodb_manager.db[self.settings.MONGO_TRANSCRIPT_COLLECTION_NAME].find(
                                    {"speaker_id": {"$in": list(relevant_speakers)}}, {"_id": 1}
                                )
                                docs_speakers = await cursor_speakers.to_list(length=None)
                                mongo_id_candidates.extend([doc["_id"] for doc in docs_speakers])
                            
                            if mongo_id_candidates:
                                unique_mongo_ids = [ObjectId(str(mid)) for mid in set(mongo_id_candidates)]
                                mongo_filter = {"_id": {"$in": unique_mongo_ids}}
                                logger.info(f"已构建 MongoDB 筛选器，包含 {len(unique_mongo_ids)} 个文档 ID。")
                            else:
                                logger.info("Neo4j 找到的上下文在 MongoDB 中没有对应文档。")
                        else:
                            logger.info("Neo4j 未找到特定上下文，正在执行全局向量搜索。")
                    else:
                        logger.info("未构建 Cypher 查询，正在执行全局向量搜索。")
            # 修复：在这里正确地捕获 Neo4j 异常
            # 你的文件开头需要添加：from neo4j.exceptions import ServiceUnavailable, Neo4jError, ClientError
            except (ServiceUnavailable, Neo4jError, ClientError) as e:
                logger.error(f"Neo4j 数据库连接失败: {e}", exc_info=True)
                logger.info("Neo4j 查询失败，正在执行全局向量搜索。")
                mongo_filter = None
            except Exception as e:
                logger.error(f"Neo4j 混合查询部分失败: {e}", exc_info=True)
                logger.info("Neo4j 查询失败，正在执行全局向量搜索。")
                mongo_filter = None 
        else:
            logger.info("在查询中未识别出人物或话题，或 Neo4j 未连接，正在执行全局向量搜索。")
        
        final_results = await self._query_milvus_and_mongo(query_text, top_k=top_k_milvus, mongo_filter=mongo_filter)
        
        return final_results

    async def _get_documents_by_speaker_id(self, speaker_id: str) -> list:
        """
        根据发言人ID从Milvus中检索所有文档。
        Args:
            speaker_id (str): 发言人ID。
        Returns:
            list: 包含该发言人所有文档的列表。
        """
        logger.info(f"正在根据发言人ID '{speaker_id}' 检索所有会议文档...")
        try:
            # 使用 meeting_milvus_manager 的 query_data 方法
            # 注意：speaker_id 字段在 Schema 中应该是可过滤的 (VARCHAR)
            query_expr = f"speaker_id == '{speaker_id}'"
            
            # 确保 meeting_milvus_manager 已初始化
            if not self.meeting_milvus_manager:
                 logger.error("Meeting Milvus Manager 未初始化")
                 return []

            result_docs = await self.meeting_milvus_manager.query_data(
                expr=query_expr,
                output_fields=["meeting_id", "speaker_id", "timestamp", "text"]
            )
            logger.info(f"为发言人ID '{speaker_id}' 找到 {len(result_docs)} 个文档。")
            return result_docs
        except Exception as e:
            logger.error(f"根据发言人ID检索文档失败: {e}", exc_info=True)
            return []


    async def get_answer_from_llm(self, question: str) -> AsyncIterator[str]:
        """
        使用 RAG 机制（混合查询）检索上下文，并流式传输 LLM 的回答。
        新增逻辑：如果问题中包含发言人ID，则优先检索该发言人的所有文档。
        Args:
            question (str): 用户的提问。
        Returns:
            AsyncIterator[str]: 回答的块流。
        """
        if not self.llm_model or not self.llm_model.is_model_loaded():
            logger.error("LLM 模型服务未初始化或未加载。")
            yield "抱歉，LLM 模型未准备好回答您的问题。"
            return

        logger.info(f"后端协调器: 正在为问题 '{question}' 执行 RAG 问答 (流式传输)...")

        # 1. 优先检查问题中是否包含声纹用户名
        speaker_id_pattern = r"(未知用户_\w{8}|新用户_\w{8})"
        match = re.search(speaker_id_pattern, question)
        
        context_str = ""
        retrieved_documents = []

        if match:
            speaker_id = match.group(1)
            logger.info(f"在问题中检测到发言人ID: {speaker_id}，正在进行定向检索。")
            
            # 1.1. 针对特定发言人ID进行检索
            # 假设 _get_documents_by_speaker_id 是一个异步方法，它会正确返回文档列表
            speaker_documents = await self._get_documents_by_speaker_id(speaker_id)
            
            if not speaker_documents:
                logger.warning(f"未找到发言人 '{speaker_id}' 的任何文档。")
                yield f"抱歉，我没有找到发言人 `{speaker_id}` 的任何会议记录。请问您是否想查询其他用户？"
                return
            
            # 1.2. 构建专门针对该发言人的上下文 (使用列表推导式简化)
            doc_parts = [
                f"## 文档 {i+1} (会议: {doc.get('meeting_id', '未知会议')}, 时间: {doc.get('timestamp', '未知时间')})\n{doc.get('text', '')}"
                for i, doc in enumerate(speaker_documents)
            ]
            context_str = f"以下是发言人 '{speaker_id}' 在多个会议中的发言记录。\n\n" + "\n\n".join(doc_parts)
            
            # 1.3. 构造LLM提示
            system_prompt = f"你是一个智能会议问答助手。请根据提供的会议记录，总结和修饰发言人 '{speaker_id}' 的发言内容，并回答用户的问题。如果信息不足，请明确说明。"
            user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

        else: # 2. 如果问题中没有发言人ID，则执行常规的 RAG 流程
            try:
                # 2.1. 执行我们修复过的混合查询
                retrieved_documents = await self._hybrid_query(question, top_k_milvus=self.settings.RAG_TOP_K)
                logger.info(f"常规 RAG 已检索到 {len(retrieved_documents)} 个文档。")
            except Exception as e:
                logger.error(f"RAG 文档检索失败: {e}", exc_info=True)
                yield f"文档检索失败，无法回答您的问题: {str(e)}"
                return

            if not retrieved_documents:
                logger.warning("未检索到相关文档，LLM 将尝试基于通用知识进行回答或声明信息不足。")
                yield "抱歉，我无法找到相关的会议信息来回答您的问题。如果您想提一个通用问题，请提供更多细节。"
                return
            
            # 2.2. 构建常规 RAG 上下文 (使用列表推导式简化)
            doc_parts = []
            for i, doc in enumerate(retrieved_documents):
                timestamp_dt = doc.get('timestamp')
                if isinstance(timestamp_dt, str):
                    try:
                        timestamp_dt = datetime.fromisoformat(timestamp_dt)
                    except ValueError:
                        timestamp_dt = None

                timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M') if timestamp_dt else "未知时间"

                doc_parts.append(
                    f"## 文档 {i+1} (会议: {doc.get('meeting_id', '未知会议')}, 发言人: {doc.get('speaker_id', '未知发言人')}, 时间: {timestamp_str})\n"
                    f"{doc.get('text', '')}"
                )
            
            context_str = "\n\n".join(doc_parts)
            
            # 2.3. 构造常规 LLM 提示
            system_prompt = "你是一个智能会议问答助手。请根据提供的会议文本片段回答用户的问题。如果信息不足，请明确说明你无法根据现有信息进行回答。请确保你的回答简洁、准确，并仅限于提供的上下文。"
            user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

        logger.debug(f"已构建的上下文:\n{context_str[:500]}...")
        
        # 3. 构建符合聊天格式的消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 4. 调用 LLM 模型，并流式返回结果
        try:
            async with self._llm_semaphore:
                # 传入 'messages' 列表，并移除不支持的 'stream' 参数
                async for chunk in self.llm_model.generate_text(messages=messages):
                    yield chunk
        except Exception as e:
            logger.error(f"LLM 流式回答生成失败: {e}", exc_info=True)
            yield f"LLM 内部服务错误: {str(e)}"

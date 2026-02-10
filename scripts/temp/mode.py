import os
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 导入 Neo4j 驱动
from neo4j import GraphDatabase

# 从独立的配置文件中导入 AppConfig
from config.conf import AppConfig 

# --- 服务初始化类 ---
class Services:
    def __init__(self, config: AppConfig):
        self.config = config
        self.milvus_collection = None
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection = None
        self.bge_tokenizer = None
        self.bge_model = None
        self.neo4j_driver = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_milvus()
        self._init_mongodb()
        self._init_bge_model()
        self._init_neo4j() #

    def _init_milvus(self):
        try:
            connections.connect(
                alias="default", 
                host=self.config.MILVUS_HOST, 
                port=self.config.MILVUS_PORT,
                user=self.config.MILVUS_USER,
                password=self.config.MILVUS_PASSWORD
            )
            print(f"Milvus 连接成功: {self.config.MILVUS_HOST}:{self.config.MILVUS_PORT}")

            if not utility.has_collection(self.config.MILVUS_COLLECTION_NAME):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="mongo_id", dtype=DataType.VARCHAR, max_length=256), 
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.MILVUS_DIMENSION)
                ]
                schema = CollectionSchema(fields, self.config.MILVUS_COLLECTION_NAME)
                self.milvus_collection = Collection(self.config.MILVUS_COLLECTION_NAME, schema)
                
                print(f"创建 Milvus 集合 '{self.config.MILVUS_COLLECTION_NAME}' 并创建索引...")
                self.milvus_collection.create_index(
                    field_name="embedding", 
                    index_params=self.config.MILVUS_INDEX_PARAMS
                )
                self.milvus_collection.load() 
                print("Milvus 集合和索引创建并加载完成。")
            else:
                print(f"Milvus 集合 '{self.config.MILVUS_COLLECTION_NAME}' 已存在，加载中...")
                self.milvus_collection = Collection(self.config.MILVUS_COLLECTION_NAME)
                self.milvus_collection.load()
                print("Milvus 集合加载完成。")

        except Exception as e:
            print(f"Milvus 连接或初始化失败: {e}")
            self.milvus_collection = None

    def _init_mongodb(self):
        try:
            self.mongo_client = MongoClient(self.config.MONGO_URI)
            self.mongo_db = self.mongo_client[self.config.MONGO_DB_NAME]
            self.mongo_collection = self.mongo_db[self.config.MONGO_COLLECTION_NAME]
            print(f"MongoDB 连接成功: {self.config.MONGO_URI}, 数据库: {self.config.MONGO_DB_NAME}, 集合: {self.config.MONGO_COLLECTION_NAME}")

        except Exception as e:
            print(f"MongoDB 连接或初始化失败: {e}")
            self.mongo_client = None
            self.mongo_db = None
            self.mongo_collection = None

    def _init_bge_model(self):
        try:
            self.bge_tokenizer = AutoTokenizer.from_pretrained(self.config.BGE_MODEL_PATH)
            self.bge_model = AutoModel.from_pretrained(self.config.BGE_MODEL_PATH).to(self.device)
            self.bge_model.eval() 
            print(f"BAAI/bge-small-zh-v1.5 模型加载成功，设备: {self.device}")
        except Exception as e:
            print(f"BAAI/bge-small-zh-v1.5 模型加载失败: {e}")
            self.bge_tokenizer = None
            self.bge_model = None
            
    def _init_neo4j(self): 
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.config.NEO4J_URI, 
                auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
            )
            self.neo4j_driver.verify_connectivity()
            print(f"Neo4j 连接成功: {self.config.NEO4J_URI}")
        except Exception as e:
            print(f"Neo4j 连接或初始化失败: {e}")
            self.neo4j_driver = None

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.bge_model is None or self.bge_tokenizer is None:
            print("错误：BGE 模型未成功加载。")
            return np.array([])
        
        encoded_input = self.bge_tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            model_output = self.bge_model(**encoded_input)
            embeddings = model_output.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def close_neo4j(self): 
        if self.neo4j_driver:
            self.neo4j_driver.close()
            print("Neo4j 连接已关闭。")

# --- 数据摄入函数 ---
def ingest_meeting_text(
    services: Services, 
    meeting_id: str, 
    speaker: str, 
    text_segment: str, 
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    处理单个会议文本片段，生成嵌入，并存储到 MongoDB、Milvus 和 Neo4j。
    """
    if services.mongo_collection is None or \
       services.milvus_collection is None or \
       services.bge_model is None:
        print("核心服务未完全初始化，无法摄入数据到 MongoDB/Milvus。")
        return {"status": "failed", "message": "Core service not ready"}

    if timestamp is None:
        timestamp = datetime.now()

    try:
        # --- MongoDB & Milvus 摄入 ---
        embedding_text = f"发言人：{speaker}。内容：{text_segment}"
        embedding = services.get_embeddings([embedding_text])[0]

        mongo_doc = {
            "meeting_id": meeting_id,
            "timestamp": timestamp,
            "speaker": speaker,
            "text": text_segment,
            "embedding_text": embedding_text,
            "embedding": embedding.tolist() 
        }
        
        insert_result = services.mongo_collection.insert_one(mongo_doc)
        mongo_doc_id = str(insert_result.inserted_id) 
        print(f"文本存储到 MongoDB: {mongo_doc_id}")

        milvus_data_mongo_ids = [mongo_doc_id]
        milvus_data_embeddings = [embedding.tolist()]
        
        services.milvus_collection.insert([milvus_data_mongo_ids, milvus_data_embeddings])
        print(f"向量存储到 Milvus，关联 MongoDB ID: {mongo_doc_id}")

        # --- Neo4j 摄入 ---
        if services.neo4j_driver:
            with services.neo4j_driver.session() as session:
                # 1. 创建或获取 Person 节点
                person_node = session.run(
                    "MERGE (p:Person {name: $speaker}) RETURN p",
                    speaker=speaker
                ).single().get("p")
                print(f"Neo4j: Person '{speaker}' 节点已处理。")

                # 2. 创建或获取 Meeting 节点
                meeting_node = session.run(
                    "MERGE (m:Meeting {id: $meeting_id}) RETURN m",
                    meeting_id=meeting_id
                ).single().get("m")
                print(f"Neo4j: Meeting '{meeting_id}' 节点已处理。")

                # 3. 创建 Person - SPOKE_IN -> Meeting 关系
                session.run(
                    """
                    MATCH (p:Person {name: $speaker})
                    MATCH (m:Meeting {id: $meeting_id})
                    MERGE (p)-[:SPOKE_IN]->(m)
                    """,
                    speaker=speaker, meeting_id=meeting_id
                )
                print(f"Neo4j: 关系 (Person)-[:SPOKE_IN]->(Meeting) 已处理。")

                # 4. 简单主题实体提取和关系创建
                predefined_topics = [
                    "线上推广方案", "短视频平台", "抖音", "快手", "预算", 
                    "产品迭代", "用户反馈模块", "市场策略会议", "预算报告"
                ]
                
                for topic_keyword in predefined_topics:
                    if topic_keyword in text_segment:
                        # 创建或获取 Topic 节点
                        topic_node = session.run(
                            "MERGE (t:Topic {name: $topic_name}) RETURN t",
                            topic_name=topic_keyword
                        ).single().get("t")
                        print(f"Neo4j: Topic '{topic_keyword}' 节点已处理。")

                        # 创建 Meeting - DISCUSSED -> Topic 关系
                        session.run(
                            """
                            MATCH (m:Meeting {id: $meeting_id})
                            MATCH (t:Topic {name: $topic_name})
                            MERGE (m)-[:DISCUSSED]->(t)
                            """,
                            meeting_id=meeting_id, topic_name=topic_keyword
                        )
                        print(f"Neo4j: 关系 (Meeting)-[:DISCUSSED]->(Topic) 已处理。")

                        # 创建 Person - MENTIONED -> Topic 关系 (如果发言人提到了这个话题)
                        session.run(
                            """
                            MATCH (p:Person {name: $speaker})
                            MATCH (t:Topic {name: $topic_name})
                            MERGE (p)-[:MENTIONED]->(t)
                            """,
                            speaker=speaker, topic_name=topic_keyword
                        )
                        print(f"Neo4j: 关系 (Person)-[:MENTIONED]->(Topic) 已处理。")
        else:
            print("Neo4j 服务未连接，跳过图数据摄入。")

        return {"status": "success", "mongo_id": mongo_doc_id, "milvus_status": "inserted", "neo4j_status": "processed"}

    except Exception as e:
        print(f"数据摄入失败: {e}")
        return {"status": "failed", "message": str(e)}

# --- 测试入口 ---
if __name__ == "__main__":
    config = AppConfig() 
    services = Services(config)

    if services.milvus_collection is None or services.mongo_collection is None or services.bge_model is None:
        print("核心服务初始化失败，请检查配置和日志。")
        exit()

    # --- 模拟前端过来的会议文本数据流  ---
    mock_meeting_data = [
        {"meeting_id": "Meet_A_20250611", "speaker": "李明", "text": "各位，今天的市场策略会议主要讨论线上推广方案。"},
        {"meeting_id": "Meet_A_20250611", "speaker": "王芳", "text": "我建议我们增加在短视频平台的投入，尤其是抖音和快手。"},
        {"meeting_id": "Meet_A_20250611", "speaker": "张三", "text": "短视频投入预算需要重新评估，我担心超出本季度预算。"},
        {"meeting_id": "Meet_A_20250611", "speaker": "李明", "text": "好的，张三，请你准备一份详细的预算报告，下周三前提交。"},
        {"meeting_id": "Meet_B_20250610", "speaker": "赵丽", "text": "上次产品迭代会议，我们决定优先开发用户反馈模块。"},
        {"meeting_id": "Meet_B_20250610", "speaker": "钱涛", "text": "用户反馈模块的开发进度目前良好，预计下月底完成初步测试。"}
    ]

    print("\n--- 清空 Milvus 和 MongoDB 数据 (重要！为了重新测试请务必执行) ---")
    try:
        if utility.has_collection(config.MILVUS_COLLECTION_NAME):
            utility.drop_collection(config.MILVUS_COLLECTION_NAME)
            print(f"Milvus 集合 '{config.MILVUS_COLLECTION_NAME}' 已清空。")
        services._init_milvus() 
        services.mongo_collection.delete_many({})
        print(f"MongoDB 集合 '{config.MONGO_COLLECTION_NAME}' 已清空。")
        
        # 清空 Neo4j 数据 (删除所有节点和关系)
        if services.neo4j_driver:
            with services.neo4j_driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("Neo4j 数据库已清空。")
        else:
            print("Neo4j 服务未连接，跳过清空。")

    except Exception as e:
        print(f"清空数据失败：{e}")
        exit("清空数据失败，请检查数据库状态和权限。")


    services.milvus_collection.load() # 确保加载 Milvus 集合

    for i, data in enumerate(mock_meeting_data):
        print(f"\n--- 摄入第 {i+1} 条数据 ---")
        result = ingest_meeting_text(
            services,
            meeting_id=data["meeting_id"],
            speaker=data["speaker"],
            text_segment=data["text"]
        )
        print(f"摄入结果: {result}")
        time.sleep(0.05) 

    print("\n--- 所有模拟数据摄入完成 ---")
    print(f"\nMilvus 中 '{config.MILVUS_COLLECTION_NAME}' 集合实体数量: {services.milvus_collection.num_entities}")
    print(f"MongoDB 中 '{config.MONGO_COLLECTION_NAME}' 集合文档数量: {services.mongo_collection.count_documents({})} ")
    
    print("\n--- Milvus Flushing... ---")
    services.milvus_collection.flush() 
    print("--- Milvus Flushed. ---")

    print("\n数据摄入服务运行完毕。请运行 'query_service.py' 进行查询验证。")
    services.close_neo4j() 
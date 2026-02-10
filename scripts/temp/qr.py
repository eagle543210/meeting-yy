import os
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymongo import MongoClient
from bson.objectid import ObjectId 
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from neo4j import GraphDatabase
from llama_cpp import Llama
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

        self._init_mongodb()
        self._init_neo4j()
        self._init_milvus()
        self._init_bge_model()

    def _init_milvus(self):
        try:
            connections.connect(
                alias="default", 
                host=self.config.MILVUS_HOST, 
                port=self.config.MILVUS_PORT,
                user=self.config.MILVUS_USER,
                password=self.config.MILVUS_PASSWORD
            )
            print(f"✅ Milvus 连接成功: {self.config.MILVUS_HOST}:{self.config.MILVUS_PORT}")

            if utility.has_collection(self.config.MILVUS_COLLECTION_NAME):
                self.milvus_collection = Collection(self.config.MILVUS_COLLECTION_NAME)
                self.milvus_collection.load() 
                print(f"✅ Milvus 集合 '{self.config.MILVUS_COLLECTION_NAME}' 加载完成。")
            else:
                print(f"❌ 错误: Milvus 集合 '{self.config.MILVUS_COLLECTION_NAME}' 不存在。请先运行 ingestion_service.py 填充数据。")
                self.milvus_collection = None 

        except Exception as e:
            print(f"❌ Milvus 连接或初始化失败: {e}")
            self.milvus_collection = None

    def _init_mongodb(self):
        try:
            self.mongo_client = MongoClient(self.config.MONGO_URI)
            self.mongo_db = self.mongo_client[self.config.MONGO_DB_NAME]
            self.mongo_collection = self.mongo_db[self.config.MONGO_COLLECTION_NAME]
            # 尝试执行一个操作以验证连接
            self.mongo_client.admin.command('ping')
            print(f"✅ MongoDB 连接成功: {self.config.MONGO_URI}, 数据库: {self.config.MONGO_DB_NAME}, 集合: {self.config.MONGO_COLLECTION_NAME}")

        except Exception as e:
            print(f"❌ MongoDB 连接或初始化失败: {e}")
            self.mongo_client = None
            self.mongo_db = None
            self.mongo_collection = None

    def _init_bge_model(self):
        try:
            self.bge_tokenizer = AutoTokenizer.from_pretrained(self.config.BGE_MODEL_PATH)
            self.bge_model = AutoModel.from_pretrained(self.config.BGE_MODEL_PATH).to(self.device)
            self.bge_model.eval() 
            print(f"✅ BAAI/bge-small-zh-v1.5 模型加载成功，设备: {self.device}")
        except Exception as e:
            print(f"❌ BAAI/bge-small-zh-v1.5 模型加载失败: {e}")
            self.bge_tokenizer = None
            self.bge_model = None
            
    def _init_neo4j(self): 
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.config.NEO4J_URI, 
                auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
            )
            self.neo4j_driver.verify_connectivity()
            print(f"✅ Neo4j 连接成功: {self.config.NEO4J_URI}")
        except Exception as e:
            print(f"❌ Neo4j 连接或初始化失败: {e}")
            self.neo4j_driver = None

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.bge_model is None or self.bge_tokenizer is None:
            print("❌ 错误：BGE 模型未成功加载。")
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

# --- 辅助函数：从查询中提取人物和主题 ---
def extract_entities_from_query(query: str) -> Dict[str, List[str]]:
    persons = []
    topics = []
    
    # 预定义的人物和主题列表 (在实际应用中，建议从数据库或配置文件中动态加载)
    all_persons = ["李明", "王芳", "张三", "赵丽", "钱涛"]
    all_topics = [
        "线上推广方案", "短视频平台", "抖音", "快手", "预算", 
        "产品迭代", "用户反馈模块", "市场策略会议", "预算报告"
    ]

    for p in all_persons:
        if p in query:
            persons.append(p)
    for t in all_topics:
        if t in query:
            topics.append(t)
            
    return {"persons": persons, "topics": topics}


# --- 通用 Milvus 查询函数 ---
def query_milvus_and_mongo(
    services: Services, 
    query_text: str, 
    top_k: int = 5,
    mongo_filter: Optional[Dict[str, Any]] = None 
) -> List[Dict[str, Any]]:
    """
    根据用户问题查询 Milvus 获取相关文本，并从 MongoDB 检索原始数据。
    可以传入 mongo_filter 来限制搜索范围。
    """
    if services.milvus_collection is None or services.mongo_collection is None or services.bge_model is None:
        print("❌ 服务未完全初始化，无法执行查询。")
        return []

    print(f"\n--- 执行 Milvus 向量查询：'{query_text}' ---")
    try:
        query_embedding = services.get_embeddings([query_text])[0]

        milvus_expr = ""
        # 修复：确保 mongo_filter 存在且包含有效的 ID 列表
        if mongo_filter and "_id" in mongo_filter and "$in" in mongo_filter["_id"] and mongo_filter["_id"]["$in"]:
            mongo_ids_list = [str(obj_id) for obj_id in mongo_filter["_id"]["$in"]]
            milvus_expr = f"mongo_id in {mongo_ids_list}" 
            print(f"   ✅ Milvus 过滤表达式: {milvus_expr}")

        search_params = {
            "data": [query_embedding.tolist()],
            "anns_field": "embedding",
            "param": {"metric_type": services.config.MILVUS_INDEX_PARAMS["metric_type"], 
                      "params": {"nprobe": 10}}, 
            "limit": top_k,
            "expr": milvus_expr, 
            "output_fields": ["mongo_id"]
        }
        
        results = services.milvus_collection.search(**search_params)
        
        retrieved_mongo_ids = []
        for hit in results[0]: 
            retrieved_mongo_ids.append(hit.entity.get('mongo_id'))

        if not retrieved_mongo_ids:
            print("   ⚠️ 未从 Milvus 检索到相关结果。")
            return []

        mongo_query_results = list(services.mongo_collection.find(
            {"_id": {"$in": [ObjectId(mid) for mid in retrieved_mongo_ids]}}
        ))
        
        mongo_doc_map = {str(doc['_id']): doc for doc in mongo_query_results}
        final_results = []
        for mid in retrieved_mongo_ids:
            if mid in mongo_doc_map:
                final_results.append(mongo_doc_map[mid])
            else:
                print(f"   ❌ 警告：MongoDB 中未找到 ID 为 {mid} 的文档。")

        return final_results

    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return []

# --- 混合查询函数 (结合 Neo4j 和 Milvus/MongoDB) ---
def hybrid_query(
    services: Services, 
    query_text: str, 
    top_k_milvus: int = 5
) -> List[Dict[str, Any]]:
    """
    执行混合查询：
    1. 尝试从查询中提取人物和主题。
    2. 如果提取到实体，则首先查询 Neo4j 获取相关上下文。
    3. 利用 Neo4j 结果作为过滤器，执行 Milvus 向量搜索。
    4. 如果未提取到实体，则直接执行 Milvus 向量搜索。
    5. 从 MongoDB 检索最终文本。
    """
    print(f"\n--- 启动混合查询：'{query_text}' ---")
    extracted_entities = extract_entities_from_query(query_text)
    persons = extracted_entities.get("persons", [])
    topics = extracted_entities.get("topics", [])
    
    mongo_filter = None
    
    if services.neo4j_driver and (persons or topics):
        print("   识别到查询中的人物或主题，尝试从 Neo4j 获取上下文...")
        with services.neo4j_driver.session() as session:
            cypher_query_parts = []
            params = {}

            if persons:
                cypher_query_parts.append("""
                    MATCH (p:Person)-[:SPOKE_IN]->(m:Meeting) WHERE p.name IN $persons_list RETURN m.id AS context_id
                    UNION
                    MATCH (p:Person)-[:MENTIONED]->(t:Topic) WHERE p.name IN $persons_list RETURN t.name AS context_id
                """)
                params["persons_list"] = persons

            if topics:
                cypher_query_parts.append("""
                    MATCH (m:Meeting)-[:DISCUSSED]->(t:Topic) WHERE t.name IN $topics_list RETURN m.id AS context_id
                    UNION
                    MATCH (p:Person)-[:MENTIONED]->(t:Topic) WHERE t.name IN $topics_list RETURN p.name AS context_id
                """)
                params["topics_list"] = topics
            
            cypher_query = " UNION ".join(cypher_query_parts)
            
            if cypher_query:
                print(f"   执行 Cypher 查询: {cypher_query}")

                neo4j_results = session.run(cypher_query, **params)
                
                relevant_meeting_ids = set()
                relevant_speakers = set()
                
                # 在 Neo4j 结果中查找匹配的会议 ID 和发言人
                distinct_meeting_ids_in_mongo = services.mongo_collection.distinct("meeting_id")
                distinct_speakers_in_mongo = services.mongo_collection.distinct("speaker")

                for record in neo4j_results:
                    context_id = record.get('context_id')
                    if context_id:
                        if context_id in distinct_meeting_ids_in_mongo:
                                relevant_meeting_ids.add(context_id)
                        elif context_id in distinct_speakers_in_mongo:
                                relevant_speakers.add(context_id)

                print(f"   Neo4j 发现相关会议ID: {relevant_meeting_ids}")
                print(f"   Neo4j 发现相关发言人: {relevant_speakers}")

                mongo_id_candidates = []
                if relevant_meeting_ids:
                    docs = services.mongo_collection.find({"meeting_id": {"$in": list(relevant_meeting_ids)}}, {"_id": 1})
                    mongo_id_candidates.extend([doc["_id"] for doc in docs])
                if relevant_speakers:
                    docs = services.mongo_collection.find({"speaker": {"$in": list(relevant_speakers)}}, {"_id": 1})
                    mongo_id_candidates.extend([doc["_id"] for doc in docs])
                
                if mongo_id_candidates:
                    unique_mongo_ids = [ObjectId(str(mid)) for mid in set(mongo_id_candidates)]
                    mongo_filter = {"_id": {"$in": unique_mongo_ids}}
                    print(f"   ✅ 已构建 MongoDB 过滤条件，包含 {len(unique_mongo_ids)} 个文档ID。")
                else:
                    print("   ⚠️ Neo4j 未能找到具体上下文，将执行全局向量搜索。")
            else:
                 print("   ⚠️ 未构建 Cypher 查询，将执行全局向量搜索。")
    else:
        print("   未识别到查询中的人物或主题，或者 Neo4j 未连接，将执行全局向量搜索。")
    
    final_results = query_milvus_and_mongo(services, query_text, top_k=top_k_milvus, mongo_filter=mongo_filter)
    
    return final_results


# --- LLM 推理函数 ---
MODEL_PATH = "./models/gguf/internlm2_5-1_8b-chat-q8_0.gguf" 

llm = None 

def initialize_llm(model_path: str):
    global llm
    if llm is None:
        try:
            print(f"✅ 正在加载 LLM 模型: {model_path}...")
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=0,       # 强制使用CPU
                n_ctx=4096,           # 上下文窗口大小
                n_threads=os.cpu_count(), # 使用所有CPU核心
                verbose=False         # 不打印llama.cpp的详细日志
            )
            print("✅ LLM 模型加载成功！")
        except Exception as e:
            print(f"❌ LLM 模型加载失败: {e}")
            llm = None
    return llm

def generate_answer_with_llm(user_query: str, retrieved_documents: List[Dict[str, Any]]) -> str:
    """
    使用LLM根据用户问题和检索到的文档流式生成答案。
    """
    if llm is None:
        print("❌ LLM 模型未加载，无法生成答案。")
        return "LLM 模型未加载，无法生成答案。"

    if not retrieved_documents:
        print("⚠️ 很抱歉，我未能找到相关的会议信息来回答您的问题。")
        return "很抱歉，我未能找到相关的会议信息来回答您的问题。如果您想问通用问题，请确保提供上下文或修改问题。"

    context_str = ""
    for i, doc in enumerate(retrieved_documents):
        context_str += f"## 文档 {i+1} (会议: {doc.get('meeting_id')}, 发言人: {doc.get('speaker')}, 时间: {doc.get('timestamp').strftime('%Y-%m-%d %H:%M')})\n"
        context_str += f"{doc.get('text')}\n\n"

    system_prompt = "你是一个智能会议问答助手，请根据提供的会议文本片段回答用户的问题。如果信息不足，请明确说明你无法根据现有信息回答。请确保你的回答简洁、准确，并仅限于提供的上下文。"
    user_prompt = f"{context_str}用户问题: {user_query}\n\n请根据上述文档，简洁明了地回答用户问题，不要添加额外信息。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print("\n--- 正在调用 LLM 流式生成答案... ---")
    full_answer = ""
    try:
        stream_response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>"],
            stream=True 
        )
        
        print("\n--- LLM 生成的最终答案 ---")
        for chunk in stream_response:
            if "content" in chunk["choices"][0]["delta"]:
                token = chunk["choices"][0]["delta"]["content"]
                print(token, end="", flush=True) 
                full_answer += token
        print("\n" + "-" * 50) 
        return full_answer.strip()

    except Exception as e:
        print(f"❌ LLM 生成答案失败: {e}")
        return "很抱歉，LLM 在生成答案时遇到问题。"


# --- 主程序入口 ---
if __name__ == "__main__":
    config = AppConfig() 
    services = Services(config)

    if services.milvus_collection is None or \
       services.mongo_collection is None or \
       services.bge_model is None:
        print("核心检索服务初始化失败，请检查配置和日志。")
        exit()

    llm_instance = initialize_llm(MODEL_PATH)
    if llm_instance is None:
        print("LLM 模型加载失败，请检查模型路径或尝试其他量化版本。")
        exit()

    print("\n\n--- 欢迎来到会议智能问答系统 ---")
    print("您可以选择查询模式，输入问题，LLM 将基于检索到的信息为您生成答案。")
    print("-" * 50)

    try:
        while True:
            print("\n请选择查询模式：")
            print("1. 仅使用 **向量搜索 (Milvus)** (适合语义相似度查询)")
            print("2. 使用 **混合搜索 (Neo4j + Milvus)** (适合包含人物/主题的关系型查询)")
            print("输入 'exit' 或 'quit' 退出。")
            
            choice = input("你的选择 (1/2/exit/quit): ").strip().lower()

            if choice in ['exit', 'quit']:
                print("退出问答系统。")
                break
            
            user_query = ""
            retrieved_docs = []

            if choice == '1':
                user_query = input("请输入你的向量查询问题: ").strip()
                if not user_query:
                    print("问题不能为空，请重新输入。")
                    continue
                print("执行仅 Milvus 向量查询...")
                retrieved_docs = query_milvus_and_mongo(services, user_query, top_k=5) 
            elif choice == '2':
                user_query = input("请输入你的混合查询问题 (例如：'李明和赵丽都说了些啥', '关于预算的会议'): ").strip()
                if not user_query:
                    print("问题不能为空，请重新输入。")
                    continue
                print("执行混合查询 (Neo4j + Milvus)...")
                retrieved_docs = hybrid_query(services, user_query, top_k_milvus=5) 
            else:
                print("无效的选择，请重新输入。")
                continue

            print("\n--- 检索到的相关文本片段：---")
            if not retrieved_docs:
                print("   (无结果)")
            else:
                for doc in retrieved_docs:
                    print(f"   - 会议: {doc.get('meeting_id')}, 发言人: {doc.get('speaker')}, 文本: '{doc.get('text')}'")
            print("-" * 50) 
            
            final_answer = generate_answer_with_llm(user_query, retrieved_docs)
            print("-" * 50)

    finally:
        services.close_neo4j() 
        if llm: 
            del llm
        print("\n--- 问答系统运行结束 ---")
import asyncio
from pymilvus import connections, list_collections, Collection, utility, DataType
import math
import random

async def get_all_milvus_data():
    """
    连接到 Milvus 实例，并尝试使用向量搜索来获取所有集合的数据。
    同时，针对 'meeting_embeddings' 集合执行一个具体的向量相似度查询。
    """
    # --- Milvus 连接配置 ---
    # 请根据你的实际情况修改以下连接参数
    host = "127.0.0.1"  # Milvus 服务器地址
    port = "19530"     # Milvus 服务器端口
    alias = "default"  # 连接别名

    # 尝试连接到 Milvus
    try:
        print("正在连接到 Milvus...")
        connections.connect(alias, host=host, port=port)
        print("连接成功！")
    except Exception as e:
        print(f"连接 Milvus 失败: {e}")
        return

    # --- 获取所有集合 ---
    try:
        collection_names = list_collections()
        print(f"\n找到以下集合: {collection_names}")
    except Exception as e:
        print(f"获取集合列表失败: {e}")
        return

    all_milvus_data = {}

    # --- 遍历每个集合获取详细信息和数据 ---
    for name in collection_names:
        print(f"\n--- 正在处理集合: '{name}' ---")
        collection = Collection(name)

        # 获取集合字段信息 (Schema)
        fields_info = []
        vector_field_name = None
        vector_dim = 0
        output_fields = []
        
        vector_types = [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]
        
        for field in collection.schema.fields:
            field_details = {
                "name": field.name,
                "data_type": field.dtype,
                "is_primary": field.is_primary,
                "is_vector": field.dtype in vector_types
            }
            fields_info.append(field_details)
            output_fields.append(field.name)

            if field.dtype in vector_types:
                vector_field_name = field.name
                vector_dim = field.params['dim']
            
        print("集合字段信息:")
        for field in fields_info:
            print(f"  - 字段名: {field['name']}, 数据类型: {field['data_type']}, 是否主键: {field['is_primary']}")

        # 确保集合已加载到内存中
        if not utility.has_collection(name) or utility.load_state(name) != 'Loaded':
             print("集合未加载，正在加载到内存...")
             collection.load()
        
        # --- 根据集合类型执行不同操作 ---
        if name == "meeting_embeddings":
            # --- 针对 meeting_embeddings 集合执行一个具体的相似度查询 ---
            print(f"\n>>> 正在针对 '{name}' 集合执行一个具体的向量查询...")
            try:
                if not vector_field_name:
                    print(f"错误: 集合 '{name}' 中没有向量字段。无法执行查询。")
                    continue
                
                # --- 重要: 请替换这里的虚拟搜索向量为你的实际会议嵌入向量 ---
                # 假设嵌入维度为 vector_dim，创建一个随机向量作为示例
                search_vector = [[random.uniform(-1, 1) for _ in range(vector_dim)]]
                top_k = 10
                
                print(f"正在使用示例向量执行 top-{top_k} 相似度查询...")

                # 执行向量搜索
                search_results = collection.search(
                    data=search_vector,
                    anns_field=vector_field_name,
                    param={},  # 使用默认搜索参数
                    limit=top_k,
                    output_fields=output_fields
                )
                
                # 打印查询结果
                print(f"\n查询成功！找到 {len(search_results[0])} 条最相似的记录:")
                for hit in search_results[0]:
                    print("-----------------------------")
                    print(f"  距离 (Distance): {hit.distance}")
                    print(f"  主键 ID: {hit.id}")
                    # 使用 hit.entity.get('字段名') 获取其他字段
                    for field_name in output_fields:
                        if field_name not in [vector_field_name, 'id']: # 避免打印向量和主键
                             print(f"  {field_name}: {hit.entity.get(field_name)}")
                print("-----------------------------")

            except Exception as e:
                print(f"查询集合 '{name}' 数据失败: {e}")
        
        else:
            # --- 对于其他集合，继续使用全量获取方法 ---
            all_records = []
            try:
                if not vector_field_name:
                    print(f"错误: 集合 '{name}' 中没有向量字段。此方法无法使用。")
                    continue
                
                total_entities = collection.num_entities
                print(f"\n集合 '{name}' 中共有 {total_entities} 条记录。")

                if total_entities == 0:
                    print("集合中没有数据。")
                    continue

                # 创建一个与向量维度匹配的虚拟搜索向量
                dummy_search_vector = [[0.0] * vector_dim]

                # 使用 search 方法来模拟全量查询
                print(f"正在使用向量搜索获取集合 '{name}' 中的所有数据...")
                
                search_results = collection.search(
                    data=dummy_search_vector,
                    anns_field=vector_field_name,
                    param={},
                    limit=total_entities,
                    output_fields=output_fields
                )
                
                all_records = [hit.entity.fields for hit in search_results[0]]
                
                print(f"成功获取所有数据，总计 {len(all_records)} 条记录。")
                all_milvus_data[name] = {
                    "schema": fields_info,
                    "data": all_records
                }
            except Exception as e:
                print(f"查询集合 '{name}' 数据失败: {e}")
        
            finally:
                # 释放集合以节省内存
                print("正在释放集合...")
                collection.release()
                print("释放完成。")
                
    return all_milvus_data

if __name__ == '__main__':
    # 调用主函数并打印结果
    asyncio.run(get_all_milvus_data())


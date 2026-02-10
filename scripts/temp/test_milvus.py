from pymilvus import connections, utility
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # 连接到 Milvus 实例
    # 默认情况下，Milvus 在 Docker 中运行时，可以通过 localhost:19530 访问
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    logger.info("✅ 成功连接到 Milvus 实例！")

    # 检查 Milvus 服务是否健康
    if utility.has_collection("test_collection"):
        utility.drop_collection("test_collection")
        logger.info("🗑️ 已删除旧的 'test_collection'。")

    collection_name = "test_collection"
    dim = 128 # 任意维度，这里只是为了测试
    
    # 创建一个简单的集合
    from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, "Test collection for Milvus connection")
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"✨ 成功创建集合 '{collection_name}'。")

    # 插入一些数据 (可选，用于进一步测试)
    # data = [[float(i) for i in range(dim)]]
    # collection.insert([{"vector": vec} for vec in data])
    # collection.flush()
    # logger.info(f"📊 集合 '{collection_name}' 中的实体数量: {collection.num_entities}")

    logger.info("测试完成，Milvus 已成功运行并可连接。")

except Exception as e:
    logger.error(f"❌ 连接 Milvus 失败或出现错误: {e}", exc_info=True)

finally:
    # 清理（可选）：断开连接，删除测试集合
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"🗑️ 已清理测试集合 '{collection_name}'。")
    except Exception as e:
        logger.warning(f"清理测试集合失败: {e}")
    connections.disconnect("default")
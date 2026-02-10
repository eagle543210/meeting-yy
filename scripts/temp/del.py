# delete_milvus_collection.py
from pymilvus import utility, connections
import os
import sys

# 尝试从你的项目路径导入 settings
# sys.path.append(os.path.join(os.path.dirname(__file__), 'config')) # 这行可能导致重复路径或混乱
# 更好的做法是确保运行脚本时，项目的根目录在 PYTHONPATH 中，或者直接写相对路径
# 考虑到你是在 M:\meeting\ 下，假设这个脚本也放在 M:\meeting\ 根目录或一个易于访问的子目录
# 为了确保能正确导入 config.settings，通常在项目根目录运行脚本

# 假设你的项目结构如下，且此脚本位于 M:\meeting\delete_milvus_collection.py
# M:\meeting\
# ├── config\
# │   └── settings.py
# └── delete_milvus_collection.py  <-- 脚本位置

# 确保能导入 settings.py
# 如果此脚本和 config 目录在同一层级，直接导入即可
# 如果此脚本在 M:\meeting\ 根目录，config 也在 M:\meeting\ 根目录，则：
try:
    from config.settings import settings
except ImportError:
    print("警告: 无法从 config.settings 导入设置。将使用默认值。")
    class DefaultSettings:
        MILVUS_HOST = "localhost"
        MILVUS_PORT = "19530"
        MILVUS_COLLECTION_NAME = "voice_prints" # <<< 统一为 voice_prints
        MILVUS_CONNECTION_ALIAS = "default" # <<< 添加 alias
    settings = DefaultSettings()

milvus_host = getattr(settings, 'MILVUS_HOST', "localhost")
milvus_port = getattr(settings, 'MILVUS_PORT', "19530")
collection_name = getattr(settings, 'MILVUS_COLLECTION_NAME', "voice_prints") # <<< 统一为 voice_prints
milvus_alias = getattr(settings, 'MILVUS_CONNECTION_ALIAS', "default") # <<< 获取 alias

print(f"尝试连接 Milvus {milvus_host}:{milvus_port} (别名: {milvus_alias})")
try:
    connections.connect(host=milvus_host, port=milvus_port, alias=milvus_alias) # <<< 传入 alias
    print("成功连接到 Milvus。")

    if utility.has_collection(collection_name, using=milvus_alias): # <<< using alias
        print(f"找到 Milvus collection '{collection_name}'。正在删除...")
        utility.drop_collection(collection_name, using=milvus_alias) # <<< using alias
        print(f"✅ Milvus collection '{collection_name}' 已成功删除。")
    else:
        print(f"Milvus collection '{collection_name}' 不存在，无需删除。")
except Exception as e:
    print(f"❌ 删除 Milvus collection 时发生错误: {e}")
finally:
    # 确保连接关闭
    connections.disconnect(alias=milvus_alias) # <<< 断开指定 alias 的连接
    print(f"Milvus 连接 '{milvus_alias}' 已断开。")
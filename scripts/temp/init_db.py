# init_db.py
import sys
from pathlib import Path
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure

# 确保项目根目录在Python路径中
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings

def init_database():
    try:
        # 连接MongoDB
        client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=5000  # 5秒连接超时
        )
        
        # 测试连接
        client.admin.command('ping')
        print("✅ 成功连接到MongoDB服务器")
        
        db = client[settings.MONGO_DB_NAME]
        
        # 初始化用户集合
        users = db.users
        users.create_index([("voiceprint_id", ASCENDING)], unique=True)
        
        # 初始化日志集合（带TTL索引）
        logs = db.meeting_logs
        logs.create_index(
            "created_at",
            expireAfterSeconds=settings.MONGO_LOG_TTL
        )
        
        # 插入示例数据
        sample_users = [
            {
                "voiceprint_id": "admin_001",
                "name": "系统管理员",
                "role": "admin",
                "created_at": datetime.utcnow(),
                "last_active": datetime.utcnow()
            },
            {
                "voiceprint_id": "manager_001",
                "name": "会议主持人",
                "role": "manager",
                "created_at": datetime.utcnow(),
                "last_active": datetime.utcnow()
            }
        ]
        
        result = users.insert_many(sample_users)
        print(f"✅ 成功插入 {len(result.inserted_ids)} 条用户数据")
        
        return True
        
    except ConnectionFailure as e:
        print(f"❌ 无法连接MongoDB: {e}")
        return False
    except OperationFailure as e:
        print(f"❌ 数据库操作失败: {e.code} - {e.details}")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {str(e)}")
        return False

if __name__ == "__main__":
    from datetime import datetime
    
    print("="*50)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 开始初始化数据库")
    print("="*50)
    
    if init_database():
        print("="*50)
        print("✅ 数据库初始化成功完成")
        print("="*50)
        sys.exit(0)
    else:
        print("="*50)
        print("❌ 数据库初始化失败")
        print("="*50)
        sys.exit(1)
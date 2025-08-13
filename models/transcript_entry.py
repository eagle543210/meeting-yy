# models\transcript_entry.py

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uuid # 导入 uuid 用于生成默认ID

class TranscriptEntry(BaseModel):
    """
    会议转录条目模型，与 MongoDB 转录集合对应。
    """
    # 修正: 添加 id 字段，并将其映射到 MongoDB 的 _id
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id", description="转录条目的唯一标识符")
    meeting_id: str = Field(..., description="会议的唯一标识符")
    client_id: str = Field(..., description="发送音频的客户端 ID")
    user_id: str = Field(..., description="识别到的发言人唯一 ID")
    speaker_id: str = Field(..., description="识别到的发言人显示名称")
    role: str = Field(..., description="发言时的用户角色")
    text: str = Field(..., description="转录的文本内容")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="转录条目创建时间 (UTC)")

    class Config:
        populate_by_name = True # 允许使用字段别名来赋值
        json_encoders = {
            datetime: lambda dt: dt.isoformat() # 序列化 datetime 为 ISO 格式字符串
        }
        arbitrary_types_allowed = True

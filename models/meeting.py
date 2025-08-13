# models\meeting.py

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

class Meeting(BaseModel):
    """
    表示一次会议的记录。
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="会议的唯一标识符")
    title: str = Field(..., description="会议的标题")
    start_time: datetime = Field(..., description="会议开始时间 (UTC)")
    end_time: Optional[datetime] = Field(None, description="会议结束时间 (UTC)，如果会议仍在进行则为 None")
    
    # 会议产出 (通常在会议结束后生成)
    transcription: Optional[str] = Field(None, description="会议的完整文本转录")
    summary: Optional[str] = Field(None, description="会议的摘要")
    action_items: Optional[List[str]] = Field(None, description="会议中识别出的行动项列表")
    
    # 参与者信息
   participants: List[str] = Field(default_factory=list, description="参与会议的用户ID列表")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="会议记录创建时间 (UTC)")

    class Config:
        # 允许从 ORM 对象（如 MongoDB 的字典）进行赋值
        from_attributes = True
        # 或者 for Pydantic v1: orm_mode = True

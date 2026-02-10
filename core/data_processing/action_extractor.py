# core/data_processing/action_extractor.py
from typing import List, Dict
import re
from datetime import datetime, timedelta

class ActionExtractor:
    def __init__(self):
        # 初始化关键词规则
        self.action_verbs = ["完成", "准备", "提交", "检查", "安排"]
        self.due_date_patterns = [
            (r"(\d+)月(\d+)日前", lambda m: datetime(datetime.now().year, int(m.group(1)), int(m.group(2)))),
            (r"(\d+)个工作日内", lambda m: datetime.now() + timedelta(days=int(m.group(1))))
        ]

    def extract(self, text: str) -> List[Dict]:
        """
        从会议文本中提取行动项
        返回格式: [{"task": "任务描述", "assignee": "负责人", "due_date": "YYYY-MM-DD"}]
        """
        actions = []
        
        # 按句子分割
        sentences = re.split(r'[。！？；]', text)
        
        for sent in sentences:
            if any(verb in sent for verb in self.action_verbs):
                # 提取负责人（简单实现：取"由XXX负责"模式）
                assignee = re.search(r"由(.+?)负责", sent)
                assignee = assignee.group(1) if assignee else "待确认"
                
                # 提取截止日期
                due_date = None
                for pattern, date_func in self.due_date_patterns:
                    match = re.search(pattern, sent)
                    if match:
                        due_date = date_func(match).strftime("%Y-%m-%d")
                        break
                
                actions.append({
                    "task": sent.strip(),
                    "assignee": assignee,
                    "due_date": due_date or (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                })
        
        return actions
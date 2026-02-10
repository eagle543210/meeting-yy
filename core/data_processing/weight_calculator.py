# core/data_processing/weight_calculator.py
from typing import Dict, List
from config.settings import settings

class WeightCalculator:
    def __init__(self):
        self.base_weights = settings.ROLE_WEIGHTS

    def compute(self, speech_data: List[Dict]) -> Dict[str, float]:
        """
        计算议题权重
        输入格式: [{
            "speaker": "发言人ID",
            "role": "角色类型",
            "duration": 发言时长(秒),
            "vote_count": 投票数
        }]
        返回格式: {"议题1": 权重值, ...}
        """
        topic_weights = {}
        
        for speech in speech_data:
            role = speech.get("role", "member")
            duration_min = speech["duration"] / 60
            votes = speech.get("vote_count", 0)
            
            # 权重计算公式
            weight = (self.base_weights.get(role, 0.5) * duration_min) + (votes * 0.2)
            
            # 累加到对应议题
            topic = speech.get("topic", "default_topic")
            topic_weights[topic] = topic_weights.get(topic, 0) + weight
        
        # 归一化处理
        max_weight = max(topic_weights.values()) if topic_weights else 1
        return {k: round(v/max_weight, 2) for k, v in topic_weights.items()}
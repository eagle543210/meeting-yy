from collections import defaultdict
from config.settings import settings

class TopicTracker:
    def __init__(self):
        self.topic_weights = defaultdict(float)
        self.role_weights = settings.ROLE_WEIGHTS

    def update_weights(self, speech_data: dict):
        """
        输入数据结构：
        {
            'topic': '产品发布',
            'speaker_role': 'host',
            'duration': 120,
            'vote_count': 3
        }
        """
        role_weight = self.role_weights.get(speech_data['speaker_role'], 0.5)
        duration_factor = speech_data['duration'] / 60  # 转换为分钟
        vote_factor = speech_data.get('vote_count', 0) * 0.2
        
        # 权重计算公式
        weight = (role_weight * duration_factor) + vote_factor
        self.topic_weights[speech_data['topic']] += weight
        
    def get_priority_topics(self, top_n=3):
        """获取权重最高的前N个议题"""
        return sorted(
            self.topic_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
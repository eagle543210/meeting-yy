# core/data_processing/__init__.py
from typing import Dict, List  # 添加类型导入
from .minute_generator import MinuteGenerator
from .topic_tracker import TopicTracker
from .action_extractor import ActionExtractor
from .weight_calculator import WeightCalculator

class DataFlowHandler:
    def __init__(self):
        self.min_gen = MinuteGenerator()
        self.topic_tracker = TopicTracker()
        self.action_ext = ActionExtractor()
        self.weight_calc = WeightCalculator()

    def process(self, meeting_text: str) -> Dict[str, object]:
        """全流程数据处理
        返回格式:
        {
            "summary": str,
            "actions": List[Dict],
            "topic_weights": Dict[str, float]
        }
        """
        summary = self.min_gen.generate(meeting_text)
        actions = self.action_ext.extract(meeting_text)
        weights = self.weight_calc.compute(
            self.topic_tracker.analyze(meeting_text)
        )
        return {
            "summary": summary,
            "actions": actions,
            "topic_weights": weights
        }

__all__ = ['DataFlowHandler']
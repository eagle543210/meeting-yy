from .scoring_engine import MeetingScorer
from .prophet_predictor import TrendPredictor
from .drools_engine import DroolsExecutor

class DecisionEngine:
    def __init__(self):
        self.scorer = MeetingScorer()
        self.predictor = TrendPredictor()
        self.rules = DroolsExecutor()
    
    def evaluate(self, meeting_data):
        # 评分
        score = self.scorer.score(meeting_data)
        
        # 预测
        forecast = self.predictor.predict(meeting_data['history'])
        
        # 规则应用
        decision = self.rules.execute({
            'score': score,
            'trend': forecast['trend'].iloc[-1],
            'participants': meeting_data.get('participant_count', 0)
        })
        
        return {
            'final_score': score * decision['weight'],
            'actions': decision['actions']
        }

__all__ = ['DecisionEngine']
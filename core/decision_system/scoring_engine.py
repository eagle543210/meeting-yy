# core/decision_system/scoring_engine.py
import joblib
import numpy as np

class MeetingScorer:
    def __init__(self):
        try:
            self.model = joblib.load('models/xgb_scorer.pkl')
        except:
            # 后备方案
            self.model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        """应急模型"""
        class DummyModel:
            def predict(self, X):
                return np.array([75 + x[0]/10 for x in X])
        return DummyModel()
    
    def score(self, features):
        input_array = [[
            features['duration'],
            features['participant_count'],
            features['topic_count']
        ]]
        return float(self.model.predict(input_array)[0])
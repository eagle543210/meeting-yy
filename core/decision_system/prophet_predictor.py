from prophet import Prophet
import pandas as pd

class TrendPredictor:
    def __init__(self):
        self.model = Prophet()
    
    def predict(self, history_data):
        """输入历史数据格式：
        [{'date': '2023-01-01', 'score': 85}, ...]
        """
        df = pd.DataFrame(history_data)
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = df['score']
        self.model.fit(df)
        future = self.model.make_future_dataframe(periods=7)
        return self.model.predict(future)
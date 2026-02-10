# core/decision_system/drools_engine.py
class DroolsExecutor:
    def execute(self, context):
        """简易规则引擎实现"""
        score = context.get('score', 0)
        participants = context.get('participants', 0)
        
        if score >= 80 and participants > 5:
            return {'weight': 1.2, 'actions': ["立即跟进","升级管理层"]}
        elif score >= 60:
            return {'weight': 1.0, 'actions': ["常规跟进"]}
        else:
            return {'weight': 0.8, 'actions': ["暂缓处理"]}
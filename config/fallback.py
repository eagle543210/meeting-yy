class FallbackModels:
    @staticmethod
    def get_spo_extractor():
        from core.knowledge_engine.spo_extractor import SPOExtractor
        return SPOExtractor()
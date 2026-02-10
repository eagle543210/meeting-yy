
from .role_manager import RoleManager


class VoiceAuthenticator:
    def __init__(self):
       
        self.role_manager = RoleManager()
        
    def analyze_audio(self, audio_path):
        embedding = self.model.extract_features(audio_path)
        return self.role_manager.verify_role(embedding)

__all__ = ['VoiceAuthenticator', 'VoiceRegistration']  # 导出新类
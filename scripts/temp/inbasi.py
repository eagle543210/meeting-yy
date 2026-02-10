import asyncio
import json
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv 
import os
import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    Query,
    Request,
    WebSocket,
    HTTPException,
    Depends,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect, WebSocketState
import inspect 

class VoiceprintService:
     def __init__(self):
        try:
                
                self.pipeline = Pipeline("pyannote/speaker-diarization@2.1", use_auth_token=True).to(device)
                self.audio_util = Audio(sample_rate=16000)
                logger.info("pyannote.audio Pipeline 和 Audio 工具初始化成功。")
        except Exception as e:
                load_error_message = f"pyannote.audio 模型加载失败: {e}. 请确认网络连接、Hugging Face 认证和模型协议是否完成，或尝试指定设备为 CPU。"
                logger.critical(load_error_message, exc_info=True)
                raise RuntimeError(load_error_message)
        self.diarization_pipeline: Optional[Pipeline] = None 
        self.embedding_model = None
        self.sample_rate = settings.VOICE_SAMPLE_RATE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_error: Optional[str] = None

        # 初始化 Milvus 管理器
        self.milvus_service = MilvusManager(settings) 
        # self.mongo_manager = MongoDBManager(settings) 

        logger.info("VoiceprintService 初始化中...")

        # 检查 pyannote.audio 是否已成功导入
        if Pipeline is None or Audio is None: # 确保 Audio 类也可用
            self._load_error = "pyannote.audio 库未成功导入 (Pipeline 或 Audio 类缺失)，说话人处理服务无法初始化。请确保已安装 pyannote.audio。"
            logger.critical(self._load_error)
            raise RuntimeError(self._load_error)

        # 检查 Hugging Face 访问令牌是否配置 (pyannote.audio 需要)
        if not settings.HUGGINGFACE_AUTH_TOKEN:
            self._load_error = "Hugging Face 访问令牌未配置 (HUGGINGFACE_AUTH_TOKEN)。说话人分离模型无法下载或加载。请在 .env 文件中设置正确的令牌。"
            logger.critical(self._load_error)
            raise RuntimeError(self._load_error)
        
        # 调用私有方法加载模型
        self._initialize_models()

        # 如果任何模型未能成功加载，抛出运行时错误以阻止服务继续初始化
        if self.diarization_pipeline is None: # 现在只检查 diarization_pipeline
            if not self._load_error:
                self._load_error = "pyannote.audio 说话人分离模型初始化失败，原因未知。"
            raise RuntimeError(self._load_error)
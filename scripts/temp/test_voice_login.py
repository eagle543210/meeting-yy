#!/usr/bin/env python3
"""
声纹登录功能测试脚本
"""

import asyncio
import numpy as np
import logging
from datetime import datetime, timezone

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_voice_login():
    """测试声纹登录功能"""
    try:
        # 模拟音频数据（1.5秒，16kHz采样率）
        sample_rate = 16000
        duration = 1.5
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        logger.info(f"生成测试音频数据: {len(audio_data)} 样本, 采样率: {sample_rate}Hz")
        
        # 模拟API请求
        import requests
        import json
        
        url = "http://localhost:8000/voice-login"
        payload = {
            "audio_data": audio_data.tolist(),
            "sample_rate": sample_rate
        }
        
        logger.info("发送声纹登录请求...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"声纹登录响应: {result}")
            
            if result.get("status") == "success":
                user = result.get("user", {})
                logger.info(f"登录成功! 用户: {user.get('name')}, 角色: {user.get('role')}")
            else:
                logger.warning(f"登录失败: {result.get('message')}")
        else:
            logger.error(f"请求失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)

if __name__ == "__main__":
    print("声纹登录功能测试")
    print("请确保后端服务已启动 (python -m uvicorn app:app --reload)")
    print("按 Enter 开始测试...")
    input()
    
    asyncio.run(test_voice_login())

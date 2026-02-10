# test_voice_biometrics.py

import sys
import os
import numpy as np
import sounddevice as sd
import time
import logging
from collections import defaultdict # 用于存储注册用户

# 确保 Python 解释器能找到你的项目模块
# 假设你的项目根目录是 M:\meeting
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Adjust path for project root

# 导入你修正好的 ECAPAWrapper
from core.voice_biometrics.ecapa_model import ECAPAWrapper
from config.settings import settings # 导入 settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量，用于存储注册用户的声纹特征
# 格式：{user_id: [embedding1, embedding2, ...]}
registered_users_features = defaultdict(list)

# 注册用户的平均声纹（用于验证和识别）
# 格式：{user_id: avg_embedding_np_array}
registered_users_avg_embeddings = {}

# 相似度阈值 (根据实际测试调整)
# 余弦相似度通常在 -1 到 1 之间，1 表示完全相同，0 表示不相关，-1 表示完全相反
# 对于声纹验证/识别，通常0.5-0.7是一个合理的起始点，具体值需根据你的数据集和模型表现微调
VERIFICATION_THRESHOLD = 0.65
IDENTIFICATION_THRESHOLD = 0.60 # 识别阈值可以略低于验证阈值

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    计算两个声纹特征向量之间的余弦相似度。
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def record_audio(duration_seconds: int, sample_rate: int) -> np.ndarray:
    """
    从麦克风录音并返回音频数据。
    """
    logger.info(f"将以 {sample_rate} Hz 采样率录音 {duration_seconds} 秒。")
    logger.info("请在提示后对着麦克风说话...")

    try:
        myrecording = sd.rec(int(duration_seconds * sample_rate),
                             samplerate=sample_rate,
                             channels=1,
                             dtype='float32')

        logger.info("开始录音...请说话...")
        sd.wait() # 等待录音完成
        logger.info("录音完成。")
        return myrecording.flatten()
    except Exception as e:
        logger.error(f"录音过程中发生错误: {e}", exc_info=True)
        return np.array([]) # 返回空数组表示失败



### **声纹注册 (Enrollment)**


def enroll_speaker(ecapa_model: ECAPAWrapper, user_id: str, num_samples: int = 3, sample_duration: int = 5):
    """
    注册一个新说话人，收集多段语音并提取声纹特征，然后计算平均特征。
    """
    logger.info(f"\n--- 开始为用户 '{user_id}' 进行声纹注册 ---")
    if user_id in registered_users_avg_embeddings:
        logger.warning(f"用户 '{user_id}' 已存在，将更新其声纹模板。")
    else:
        logger.info(f"正在注册新用户 '{user_id}'。")

    enrollment_embeddings = []
    for i in range(num_samples):
        logger.info(f"第 {i+1}/{num_samples} 段录音用于注册。")
        audio_data = record_audio(sample_duration, ecapa_model.sample_rate)
        if audio_data.size == 0:
            logger.error("录音失败，跳过当前样本。")
            continue

        embedding = ecapa_model.extract_features_from_buffer(audio_data, ecapa_model.sample_rate)
        if embedding is not None:
            enrollment_embeddings.append(embedding)
            logger.info(f"第 {i+1} 段声纹特征提取成功，维度: {embedding.shape}")
        else:
            logger.error(f"第 {i+1} 段声纹特征提取失败。")

    if not enrollment_embeddings:
        logger.error(f"❌ 用户 '{user_id}' 注册失败，未能收集到有效的声纹特征。")
        return False

    # 计算平均声纹特征作为注册模板
    avg_embedding = np.mean(enrollment_embeddings, axis=0)
    registered_users_avg_embeddings[user_id] = avg_embedding
    registered_users_features[user_id].extend(enrollment_embeddings) # 可选：保留所有原始嵌入

    logger.info(f"✅ 用户 '{user_id}' 声纹注册成功！平均特征维度: {avg_embedding.shape}")
    logger.info(f"--- 用户 '{user_id}' 声纹注册结束 ---\n")
    return True



### **声纹验证 (Verification)**


def verify_speaker(ecapa_model: ECAPAWrapper, user_id_to_verify: str, sample_duration: int = 5):
    """
    验证说话人是否为特定注册用户。
    """
    logger.info(f"\n--- 开始验证用户 '{user_id_to_verify}' ---")

    if user_id_to_verify not in registered_users_avg_embeddings:
        logger.warning(f"用户 '{user_id_to_verify}' 未注册。请先注册。")
        logger.info(f"--- 用户 '{user_id_to_verify}' 验证结束 ---\n")
        return False

    registered_embedding = registered_users_avg_embeddings[user_id_to_verify]

    logger.info("请提供您的语音进行验证。")
    audio_data = record_audio(sample_duration, ecapa_model.sample_rate)
    if audio_data.size == 0:
        logger.error("录音失败，无法进行验证。")
        logger.info(f"--- 用户 '{user_id_to_verify}' 验证结束 ---\n")
        return False

    current_embedding = ecapa_model.extract_features_from_buffer(audio_data, ecapa_model.sample_rate)

    if current_embedding is None:
        logger.error("❌ 当前语音特征提取失败，无法进行验证。")
        logger.info(f"--- 用户 '{user_id_to_verify}' 验证结束 ---\n")
        return False

    similarity = cosine_similarity(registered_embedding, current_embedding)
    logger.info(f"用户 '{user_id_to_verify}' 的验证相似度: {similarity:.4f} (阈值: {VERIFICATION_THRESHOLD:.4f})")

    if similarity >= VERIFICATION_THRESHOLD:
        logger.info(f"✅ 验证成功！说话人被确认为用户 '{user_id_to_verify}'。")
        return True
    else:
        logger.warning(f"❌ 验证失败。说话人不是用户 '{user_id_to_verify}'。")
        return False
    logger.info(f"--- 用户 '{user_id_to_verify}' 验证结束 ---\n")



### **声纹识别 (Identification)**


def identify_speaker(ecapa_model: ECAPAWrapper, sample_duration: int = 5):
    """
    识别说话人是哪个已注册用户。
    """
    logger.info(f"\n--- 开始声纹识别 ---")

    if not registered_users_avg_embeddings:
        logger.warning("没有注册用户，无法进行识别。请先注册用户。")
        logger.info(f"--- 声纹识别结束 ---\n")
        return None, 0.0

    logger.info("请提供您的语音进行识别。")
    audio_data = record_audio(sample_duration, ecapa_model.sample_rate)
    if audio_data.size == 0:
        logger.error("录音失败，无法进行识别。")
        logger.info(f"--- 声纹识别结束 ---\n")
        return None, 0.0

    current_embedding = ecapa_model.extract_features_from_buffer(audio_data, ecapa_model.sample_rate)

    if current_embedding is None:
        logger.error("❌ 当前语音特征提取失败，无法进行识别。")
        logger.info(f"--- 声纹识别结束 ---\n")
        return None, 0.0

    max_similarity = -1.0
    identified_user = None

    logger.info("正在与注册用户进行比对...")
    for user_id, registered_embedding in registered_users_avg_embeddings.items():
        similarity = cosine_similarity(registered_embedding, current_embedding)
        logger.info(f"  - 与用户 '{user_id}' 相似度: {similarity:.4f}")
        if similarity > max_similarity:
            max_similarity = similarity
            identified_user = user_id

    logger.info(f"最高相似度: {max_similarity:.4f} (匹配用户: {identified_user}) (阈值: {IDENTIFICATION_THRESHOLD:.4f})")

    if identified_user and max_similarity >= IDENTIFICATION_THRESHOLD:
        logger.info(f"✅ 识别成功！说话人被识别为用户 '{identified_user}'。")
        return identified_user, max_similarity
    else:
        logger.warning("❌ 识别失败。未找到匹配的注册用户或相似度过低。")
        return None, max_similarity
    logger.info(f"--- 声纹识别结束 ---\n")



### **主测试流程**


if __name__ == "__main__":
    # 1. 实例化 ECAPAWrapper 模型
    try:
        ecapa_model = ECAPAWrapper()
        logger.info("ECAPAWrapper 模型实例化成功。")
    except RuntimeError as e:
        logger.error(f"ECAPAWrapper 模型实例化失败: {e}")
        logger.info("测试终止。")
        sys.exit(1) # 退出程序

    # 2. 模拟声纹注册
    logger.info("\n--- 模拟声纹注册阶段 ---")
    user1_id = "Alice"
    user2_id = "Bob"

    enroll_speaker(ecapa_model, user1_id)
    time.sleep(1) # 稍作等待，防止录音设备立即被占用
    enroll_speaker(ecapa_model, user2_id)
    time.sleep(1)

    logger.info("\n--- 注册用户列表 ---")
    if registered_users_avg_embeddings:
        for user_id in registered_users_avg_embeddings.keys():
            logger.info(f"  - {user_id}")
    else:
        logger.info("目前没有注册用户。")
    logger.info("---------------------\n")


    # 3. 模拟声纹验证
    logger.info("\n--- 模拟声纹验证阶段 ---")

    # 尝试验证 Alice (Alice本人说话)
    logger.info(">> 测试验证用户 'Alice' (请 Alice 本人说话)")
    verify_speaker(ecapa_model, user1_id)
    time.sleep(1)

    # 尝试验证 Alice (Bob说话，应该失败)
    logger.info(">> 测试验证用户 'Alice' (请 Bob 说话，预期失败)")
    verify_speaker(ecapa_model, user1_id)
    time.sleep(1)

    # 尝试验证 Bob (Bob本人说话)
    logger.info(">> 测试验证用户 'Bob' (请 Bob 本人说话)")
    verify_speaker(ecapa_model, user2_id)
    time.sleep(1)

    # 4. 模拟声纹识别
    logger.info("\n--- 模拟声纹识别阶段 ---")

    # 尝试识别一个已注册用户 (Alice说话)
    logger.info(">> 测试识别说话人 (请 Alice 说话)")
    identified_user, similarity = identify_speaker(ecapa_model)
    if identified_user:
        logger.info(f"识别结果: 用户 '{identified_user}', 相似度: {similarity:.4f}")
    else:
        logger.info("识别失败。")
    time.sleep(1)

    # 尝试识别另一个已注册用户 (Bob说话)
    logger.info(">> 测试识别说话人 (请 Bob 说话)")
    identified_user, similarity = identify_speaker(ecapa_model)
    if identified_user:
        logger.info(f"识别结果: 用户 '{identified_user}', 相似度: {similarity:.4f}")
    else:
        logger.info("识别失败。")
    time.sleep(1)

    # 尝试识别一个未注册用户 (非 Alice/Bob 的人说话，应该失败或识别为相似度低的已注册用户)
    logger.info(">> 测试识别说话人 (请一个未注册用户说话，预期失败或低相似度匹配)")
    identified_user, similarity = identify_speaker(ecapa_model)
    if identified_user:
        logger.info(f"识别结果: 用户 '{identified_user}', 相似度: {similarity:.4f}")
    else:
        logger.info("识别失败。")

    logger.info("\n--- 所有声纹功能测试结束 ---")
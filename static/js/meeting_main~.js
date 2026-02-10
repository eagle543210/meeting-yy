// static/js/meeting_main.js

import { v4 as uuidv4 } from 'https://jspm.dev/uuid'; // 用于生成唯一的 meeting_id 和 client_id

// --- UI 元素获取 ---
const recordBtn = document.getElementById('record-btn');
const permissionWarning = document.getElementById('permission-warning');
const meetingStatusSpan = document.getElementById('meeting-status');
const currentSpeakerSpan = document.getElementById('current-speaker');
const speakerConfidenceSpan = document.getElementById('speaker-confidence');
const transcriptContainer = document.getElementById('transcript-container');
const emptyStateDiv = transcriptContainer.querySelector('.empty-state');
const participantsList = document.getElementById('participants-list');
const participantCountSpan = document.getElementById('participant-count');
const volumeMeter = document.getElementById('volume-meter');
const volumeText = document.getElementById('volume-text');
const volumePeak = document.getElementById('volume-peak');
const volumeAvg = document.getElementById('volume-avg');
const wsStatusSpan = document.getElementById('ws-status');
const wsQueueSpan = document.getElementById('ws-queue');
const speechTemplate = document.getElementById('speech-template');
const participantTemplate = document.getElementById('participant-template');

// --- 全局变量 ---
let mediaRecorder;
let audioChunks = [];
let websocket;
let audioContext;
let analyserNode;
let microphoneSource;
let volumePollingInterval;

let isRecording = false;
let meetingId = ''; // 当前会议ID
let clientId = uuidv4(); // 客户端唯一ID
let participants = new Map(); // Map<speakerId, {name: string, role: string, lastActive: Date, speechCount: number, element: HTMLElement}>
let currentSpeechBubble = null; // 当前正在更新的发言气泡

const WS_URL = 'ws://localhost:8000/ws/meeting/'; // 替换为你的后端 WebSocket URL

// --- 音频处理配置 ---
const AUDIO_SAMPLE_RATE = 16000; // 16kHz 采样率
const AUDIO_CHANNELS = 1; // 单声道
const AUDIO_BIT_DEPTH = 16; // 16位 PCM
const CHUNK_SIZE_MS = 200; // 每 200 毫秒发送一次音频数据

// --- 工具函数 ---

// 更新连接状态UI
function updateWsStatus(status, queueSize = 0) {
    wsStatusSpan.textContent = `连接状态: ${status}`;
    wsQueueSpan.textContent = `待发消息: ${queueSize}`;
    wsStatusSpan.className = ''; // Reset classes
    if (status === '已连接') {
        wsStatusSpan.classList.add('text-success');
    } else if (status === '连接中...') {
        wsStatusSpan.classList.add('text-warning');
    } else {
        wsStatusSpan.classList.add('text-danger');
    }
}

// 显示警告信息
function showPermissionWarning(message) {
    permissionWarning.querySelector('span').textContent = message;
    permissionWarning.classList.remove('d-none');
}

// 隐藏警告信息
function hidePermissionWarning() {
    permissionWarning.classList.add('d-none');
}

// 更新会议状态
function updateMeetingStatus(status, type = 'secondary') {
    meetingStatusSpan.textContent = status;
    meetingStatusSpan.className = `badge bg-${type}`;
}

// 更新当前发言人信息
function updateCurrentSpeaker(speakerName, confidence = 0) {
    currentSpeakerSpan.textContent = speakerName || '等待识别发言人...';
    speakerConfidenceSpan.textContent = `${confidence.toFixed(0)}%`;
}

// 更新音量指示器
function updateVolumeMeter(volume) {
    const clampedVolume = Math.min(100, Math.max(0, volume)); // 限制在0-100
    volumeMeter.style.width = `${clampedVolume}%`;
    volumeMeter.setAttribute('aria-valuenow', clampedVolume);
    volumeText.textContent = `${clampedVolume.toFixed(0)}% (${isRecording ? '激活' : '未激活'})`;

    // 渐变颜色
    if (clampedVolume > 70) {
        volumeMeter.className = 'progress-bar bg-danger volume-indicator';
    } else if (clampedVolume > 40) {
        volumeMeter.className = 'progress-bar bg-warning volume-indicator';
    } else if (clampedVolume > 0) {
        volumeMeter.className = 'progress-bar bg-success volume-indicator';
    } else {
        volumeMeter.className = 'progress-bar bg-secondary volume-indicator';
    }
}

// 更新参会人员列表
function updateParticipantListUI() {
    participantsList.innerHTML = ''; // 清空现有列表
    participantCountSpan.textContent = `${participants.size}人`;

    // 将Map转换为数组并按最后活跃时间排序（最近活跃的在前）
    const sortedParticipants = Array.from(participants.values()).sort((a, b) => b.lastActive - a.lastActive);

    sortedParticipants.forEach(p => {
        if (!p.element) {
            const clone = participantTemplate.content.cloneNode(true);
            p.element = clone.querySelector('.list-group-item');
            p.element.dataset.speakerId = p.id; // 存储 speaker ID
            p.element.querySelector('.participant-name').textContent = p.name;
            p.element.querySelector('.participant-role').textContent = p.role ? `角色: ${p.role}` : '未知角色';
        }
        p.element.querySelector('.last-active').textContent = formatTimeAgo(p.lastActive);
        p.element.querySelector('.speech-count').textContent = `发言: ${p.speechCount}次`;
        participantsList.appendChild(p.element);
    });
}

// 格式化时间显示 (例如：5分钟前)
function formatTimeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);
    if (seconds < 60) return `${seconds}秒前`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}分钟前`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}小时前`;
    const days = Math.floor(hours / 24);
    return `${days}天前`;
}

// --- WebSocket 处理 ---

function connectWebSocket() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        console.warn("WebSocket已连接，无需重复连接。");
        return;
    }

    // 生成新的 meetingId (每次点击开始录音时生成新的会议ID)
    // 或者你可以从后端接口获取一个现有的 meetingId
    meetingId = `meeting_${Date.now()}`; 
    updateWsStatus('连接中...');

    websocket = new WebSocket(`${WS_URL}${meetingId}/${clientId}`);

    websocket.onopen = () => {
        console.log('WebSocket连接已建立');
        updateWsStatus('已连接', audioChunks.length);
        // 通知后端客户端已准备就绪
        websocket.send(JSON.stringify({ type: 'client_ready', meeting_id: meetingId, client_id: clientId }));
        // 告诉后端开始录音
        websocket.send(JSON.stringify({ type: 'start_recording', meeting_id: meetingId, client_id: clientId }));
        updateMeetingStatus('录音中', 'danger');
        emptyStateDiv.classList.add('d-none'); // 隐藏空状态提示
    };

    websocket.onmessage = async (event) => {
        const message = JSON.parse(event.data);
        // console.log('收到WebSocket消息:', message);

        switch (message.type) {
            case 'meeting_init_response':
                console.log('会议初始化响应:', message.data);
                // 可以在这里处理初始会议信息
                break;
            case 'speaker_identified':
                // 当后端识别到说话人时
                const { speaker_id, speaker_name, role, is_new_speaker, confidence } = message.data;
                updateCurrentSpeaker(speaker_name, confidence);

                // 更新或添加参会人员列表
                let participant = participants.get(speaker_id);
                if (!participant) {
                    participant = {
                        id: speaker_id,
                        name: speaker_name,
                        role: role,
                        lastActive: new Date(),
                        speechCount: 0,
                        element: null // 待会创建DOM元素
                    };
                    participants.set(speaker_id, participant);
                    console.log(`新发言人加入: ${speaker_name} (${speaker_id})`);
                }
                participant.lastActive = new Date(); // 更新活跃时间
                // 如果是新识别的说话人，或其名字有更新，则更新列表项
                if (is_new_speaker || participant.name !== speaker_name || participant.role !== role) {
                    participant.name = speaker_name;
                    participant.role = role;
                    if (participant.element) { // 如果元素已存在，更新其内容
                        participant.element.querySelector('.participant-name').textContent = participant.name;
                        participant.element.querySelector('.participant-role').textContent = participant.role ? `角色: ${participant.role}` : '未知角色';
                    }
                }
                updateParticipantListUI(); // 重新渲染参会人员列表

                // 如果当前没有发言气泡或者说话人变化了，就创建一个新的
                if (!currentSpeechBubble || currentSpeechBubble.dataset.speakerId !== speaker_id) {
                    currentSpeechBubble = createSpeechBubble(speaker_name);
                    transcriptContainer.appendChild(currentSpeechBubble);
                }
                // 滚动到底部
                transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
                break;
            case 'transcription_update':
                // 实时转录文本更新
                if (currentSpeechBubble && currentSpeechBubble.dataset.speakerId === message.data.speaker_id) {
                    const contentDiv = currentSpeechBubble.querySelector('.speech-content');
                    // 使用marked.js处理Markdown（如果需要）
                    contentDiv.innerHTML = marked.parse(message.data.transcript);
                } else {
                    // 如果说话人没识别出来或者气泡不匹配，先简单显示在当前发言人处
                    // 更理想的做法是创建一个"未知发言人"的气泡
                    updateCurrentSpeaker(`发言: ${message.data.transcript.substring(0, 20)}...`);
                }
                // 确保滚动到底部，以便看到最新内容
                transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
                break;
            case 'final_transcription':
                // 最终转录结果，可以更新发言计数
                const { speaker_id: final_speaker_id, transcript: final_transcript } = message.data;
                const finalParticipant = participants.get(final_speaker_id);
                if (finalParticipant) {
                    finalParticipant.speechCount++;
                    updateParticipantListUI();
                }
                // 这里可以做一些标记，表示这个 speech bubble 的内容是最终的了
                console.log(`最终转录: ${final_speaker_id} - ${final_transcript}`);
                break;
            case 'error':
                console.error('WebSocket错误:', message.data);
                alert(`WebSocket错误: ${message.data.message}`);
                break;
            default:
                console.warn('未知消息类型:', message.type, message);
        }
    };

    websocket.onclose = (event) => {
        console.log('WebSocket连接已关闭:', event);
        updateWsStatus('断开');
        updateMeetingStatus('已结束', 'secondary');
        isRecording = false;
        recordBtn.classList.remove('btn-success');
        recordBtn.classList.add('btn-danger');
        recordBtn.innerHTML = '<i class="bi bi-mic"></i>';
        updateVolumeMeter(0);
        clearInterval(volumePollingInterval); // 停止音量检测
        if (event.code !== 1000) { // 1000 是正常关闭
            showPermissionWarning(`连接断开: ${event.reason || '未知错误'}. 请尝试重新开始。`);
        }
    };

    websocket.onerror = (error) => {
        console.error('WebSocket错误:', error);
        updateWsStatus('错误');
        updateMeetingStatus('错误', 'danger');
        showPermissionWarning('WebSocket连接错误，请检查网络或后端服务。');
        websocket.close();
    };
}

function closeWebSocket() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        // 通知后端停止录音
        websocket.send(JSON.stringify({ type: 'stop_recording', meeting_id: meetingId, client_id: clientId }));
        websocket.close();
    }
}

// --- 音频录制与发送 ---

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        hidePermissionWarning();

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        // 确保采样率匹配后端需求
        if (audioContext.sampleRate !== AUDIO_SAMPLE_RATE) {
            console.warn(`浏览器默认采样率 ${audioContext.sampleRate} 与目标 ${AUDIO_SAMPLE_RATE} 不符，可能需要重采样。`);
            // 在实际项目中，这里需要添加 AudioWorkletNode 或 Resampler 来进行重采样
            // 目前简化处理，直接使用 MediaRecorder 默认或浏览器协商的采样率
        }

        microphoneSource = audioContext.createMediaStreamSource(stream);

        // 音量监测 (Web Audio API)
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 2048;
        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        microphoneSource.connect(analyserNode);

        let maxVolume = 0;
        let volumeHistory = []; // 用于计算平均值
        const VOLUME_HISTORY_SIZE = 30; // 记录最近30个采样，每200ms采样一次，即6秒历史

        volumePollingInterval = setInterval(() => {
            analyserNode.getByteFrequencyData(dataArray);
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                sum += dataArray[i];
            }
            const averageVolume = sum / bufferLength;
            const currentVolume = Math.floor((averageVolume / 255) * 100); // 归一化到0-100

            updateVolumeMeter(currentVolume);

            maxVolume = Math.max(maxVolume, currentVolume);
            volumePeak.textContent = maxVolume;

            volumeHistory.push(currentVolume);
            if (volumeHistory.length > VOLUME_HISTORY_SIZE) {
                volumeHistory.shift();
            }
            const avgOfHistory = volumeHistory.reduce((a, b) => a + b, 0) / volumeHistory.length;
            volumeAvg.textContent = avgOfHistory.toFixed(0);

        }, 200); // 每200毫秒更新一次音量

        // MediaRecorder setup
        // IMPORTANT: MIME type should match backend's expected audio format.
        // For raw PCM data over WebSocket, usually 'audio/wav' or 'audio/webm' (with appropriate codecs) is used,
        // then backend extracts raw PCM. Or, directly send raw PCM from AudioWorklet.
        // For simplicity, we'll try common webm and let backend handle.
        const mimeType = MediaRecorder.isTypeSupported('audio/webm; codecs=opus') ? 'audio/webm; codecs=opus' : 'audio/webm';
        
        mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
                sendAudioChunks(); // 每次有新数据块时尝试发送
            }
        };

        mediaRecorder.onstop = () => {
            console.log("录音停止，清空音频数据。");
            audioChunks = [];
            // Stream tracks might need to be stopped explicitly to release mic
            stream.getTracks().forEach(track => track.stop());
            clearInterval(volumePollingInterval); // 停止音量检测
            updateVolumeMeter(0); // 重置音量显示
        };

        mediaRecorder.start(CHUNK_SIZE_MS); // 每 CHUNK_SIZE_MS 毫秒触发一次 ondataavailable
        isRecording = true;
        recordBtn.classList.remove('btn-danger');
        recordBtn.classList.add('btn-success');
        recordBtn.innerHTML = '<i class="bi bi-stop-fill"></i>'; // 停止图标
        updateMeetingStatus('录音中', 'danger');
        connectWebSocket(); // 开始录音后连接WebSocket
        currentSpeechBubble = null; // 重置当前发言气泡
        transcriptContainer.innerHTML = ''; // 清空转录显示
        emptyStateDiv.classList.add('d-none'); // 隐藏空状态提示
        participants.clear(); // 清空参会人员列表
        updateParticipantListUI(); // 更新UI

    } catch (err) {
        console.error('获取麦克风失败:', err);
        showPermissionWarning('无法获取麦克风权限，请检查浏览器设置。');
        updateMeetingStatus('错误', 'danger');
        isRecording = false;
        recordBtn.classList.remove('btn-success');
        recordBtn.classList.add('btn-danger');
        recordBtn.innerHTML = '<i class="bi bi-mic"></i>';
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.classList.remove('btn-success');
        recordBtn.classList.add('btn-danger');
        recordBtn.innerHTML = '<i class="bi bi-mic"></i>'; // 麦克风图标
        updateMeetingStatus('已结束', 'secondary');
        closeWebSocket(); // 停止录音后关闭WebSocket
        currentSpeechBubble = null;
    }
}

// 定时发送累积的音频数据
async function sendAudioChunks() {
    if (websocket && websocket.readyState === WebSocket.OPEN && audioChunks.length > 0) {
        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        try {
            // 将 Blob 转换为 ArrayBuffer 再发送
            const arrayBuffer = await blob.arrayBuffer();
            websocket.send(arrayBuffer);
            audioChunks = []; // 发送后清空
            updateWsStatus('已连接', audioChunks.length);
        } catch (e) {
            console.error("发送音频数据失败:", e);
            updateWsStatus('发送失败', audioChunks.length);
        }
    }
}

// --- UI 交互事件监听 ---

recordBtn.addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// --- 会议纪要显示处理 ---
function createSpeechBubble(speakerName, content = '') {
    const clone = speechTemplate.content.cloneNode(true);
    const speechBubble = clone.querySelector('.speech-bubble');
    const speakerHeader = speechBubble.querySelector('.speaker-name');
    const speakerTime = speechBubble.querySelector('.speaker-time');
    const speechContent = speechBubble.querySelector('.speech-content');

    speakerHeader.textContent = speakerName;
    speakerTime.textContent = new Date().toLocaleTimeString();
    speechContent.innerHTML = marked.parse(content);

    // 存储 speaker ID 到气泡上，方便后续更新
    // speaker_id 应该由后端返回
    // 这里暂时用 speakerName 作为 ID，实际应使用后端返回的 speaker_id
    speechBubble.dataset.speakerId = speakerName; 
    
    return speechBubble;
}

// 初始化时检查并加载 AudioWorklet
document.addEventListener('DOMContentLoaded', () => {
    // 调用loadAudioProcessor() 如果你需要它进行更复杂的音频处理
    // loadAudioProcessor();
    console.log("前端页面已加载。");
    updateWsStatus('断开');
    updateMeetingStatus('未开始');
    updateCurrentSpeaker('等待开始...');
    hidePermissionWarning();
});

// 导出按钮权限（根据后端返回的用户权限来控制显示）
// 这些按钮的点击事件处理逻辑会调用后端的 /generate-report 等接口
document.getElementById('export-minutes-btn').addEventListener('click', async () => {
    if (!meetingId) {
        alert("请先开始并结束会议才能导出会议纪要。");
        return;
    }
    // 假设后端有 /export-minutes/{meeting_id} 接口
    try {
        const response = await fetch(`/export-minutes/${meetingId}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `会议纪要_${meetingId}.docx`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            alert("会议纪要导出成功！");
        } else {
            const errorData = await response.json();
            alert(`导出会议纪要失败: ${errorData.detail || response.statusText}`);
        }
    } catch (error) {
        console.error("导出会议纪要请求失败:", error);
        alert("导出会议纪要请求失败，请检查网络。");
    }
});

document.getElementById('generate-report-btn').addEventListener('click', async () => {
    if (!meetingId) {
        alert("请先开始并结束会议才能生成总结报告。");
        return;
    }
    // 假设后端有 /generate-report/{meeting_id} 接口
    try {
        const response = await fetch(`/generate-report/${meetingId}`);
        if (response.ok) {
            const result = await response.json();
            if (result.status === "success") {
                alert("总结报告已成功生成，请在后端查看或等待下载链接（如果后端提供）。");
                // 如果后端直接返回PDF文件流，这里需要像导出纪要一样处理blob
            } else {
                alert(`生成总结报告失败: ${result.message}`);
            }
        } else {
            const errorData = await response.json();
            alert(`生成总结报告失败: ${errorData.detail || response.statusText}`);
        }
    } catch (error) {
        console.error("生成总结报告请求失败:", error);
        alert("生成总结报告请求失败，请检查网络。");
    }
});

// 模拟权限控制，实际应从后端获取当前用户权限
// function applyPermissions(permissions) {
//     document.querySelectorAll('[data-permission]').forEach(el => {
//         const requiredPermission = el.dataset.permission;
//         if (!permissions.includes(requiredPermission)) {
//             el.style.display = 'none'; // 或者 el.disabled = true;
//         } else {
//             el.style.display = ''; // Show if permission exists
//         }
//     });
// }
// 示例：假设用户有这些权限
// applyPermissions(['export_minutes', 'generate_reports', 'edit_roles', 'export_user_speech']);

// 注册新用户/更新角色功能（需要与后端 /register-voice 或 /update-user-role 接口联动）
// 这里只提供一个UI骨架和简单的事件监听，具体逻辑需要完善
const userSelect = document.getElementById('user-select');
const roleSelect = document.getElementById('role-select');
const updateRoleBtn = document.getElementById('update-role-btn');

// 示例：动态填充用户选择框 (从 participants Map 中获取)
function populateUserSelect() {
    userSelect.innerHTML = '<option value="">-- 请选择用户 --</option>';
    participants.forEach(p => {
        const option = document.createElement('option');
        option.value = p.id; // 使用 speaker_id 作为 value
        option.textContent = p.name;
        userSelect.appendChild(option);
    });
}

// 监听用户选择变化，自动填充角色
userSelect.addEventListener('change', () => {
    const selectedSpeakerId = userSelect.value;
    const participant = participants.get(selectedSpeakerId);
    if (participant && participant.role) {
        roleSelect.value = participant.role;
    } else {
        roleSelect.value = 'member'; // 默认角色
    }
});

// 更新角色按钮点击事件
updateRoleBtn.addEventListener('click', async () => {
    const selectedSpeakerId = userSelect.value;
    const newRole = roleSelect.value;

    if (!selectedSpeakerId) {
        alert("请选择一个用户来更新角色。");
        return;
    }

    // 假设后端有 /update-user-role 接口，接收 speaker_id 和 role
    try {
        const response = await fetch('/update-user-role', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ speaker_id: selectedSpeakerId, role: newRole })
        });

        if (response.ok) {
            const result = await response.json();
            if (result.status === "success") {
                alert(`用户 ${participants.get(selectedSpeakerId).name} 的角色已更新为 ${newRole}！`);
                // 更新前端的 participants Map 并刷新 UI
                const participant = participants.get(selectedSpeakerId);
                if (participant) {
                    participant.role = newRole;
                    updateParticipantListUI();
                }
            } else {
                alert(`更新角色失败: ${result.message}`);
            }
        } else {
            const errorData = await response.json();
            alert(`更新角色请求失败: ${errorData.detail || response.statusText}`);
        }
    } catch (error) {
        console.error("更新角色请求失败:", error);
        alert("更新角色请求失败，请检查网络。");
    }
});

// 确保在参会人员列表更新时，同时更新用户选择框
const originalUpdateParticipantListUI = updateParticipantListUI;
updateParticipantListUI = () => {
    originalUpdateParticipantListUI();
    populateUserSelect(); // 每次更新参会人员列表后，刷新用户选择框
};
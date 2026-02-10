// M:\meeting\static\js\meeting_main.js

// 常量和DOM元素引用
const WS_BASE_URL = "ws://localhost:8000/ws/meeting"; // WebSocket基础URL
const API_BASE_URL = "http://localhost:8000"; // API基础URL

const recordBtn = document.getElementById('record-btn');
const meetingStatusSpan = document.getElementById('meeting-status');
const currentSpeakerSpan = document.getElementById('current-speaker');
const speakerConfidenceSpan = document.getElementById('speaker-confidence');
const transcriptContainer = document.getElementById('transcript-container');
const emptyStateDiv = transcriptContainer.querySelector('.empty-state');
const liveSummaryDiv = document.getElementById('live-summary');
const wsStatusSpan = document.getElementById('ws-status');
const wsQueueSpan = document.getElementById('ws-queue');
const volumeMeter = document.getElementById('volume-meter');
const volumeText = document.getElementById('volume-text');
const volumePeak = document.getElementById('volume-peak');
const volumeAvg = document.getElementById('volume-avg');
const permissionWarning = document.getElementById('permission-warning');
const participantsList = document.getElementById('participants-list');
const speechTemplate = document.getElementById('speech-template');
const participantTemplate = document.getElementById('participant-template');

// 管理面板相关元素
const adminPanel = document.querySelector('.admin-panel'); // 权限管理卡片
const userSelect = document.getElementById('user-select');
const roleSelect = document.getElementById('role-select');
const updateRoleBtn = document.getElementById('update-role-btn');

// 导出按钮
const exportMinutesBtn = document.getElementById('export-minutes-btn');
const exportSpeechesBtn = document.getElementById('export-speeches-btn'); // 新增：导出用户发言按钮
const generateReportBtn = document.getElementById('generate-report-btn');

// 全局变量
let websocket = null;
let mediaStream = null;
let audioContext = null;
let audioWorkletNode = null;
let audioInput = null;
let isRecording = false;
const SAMPLE_RATE = 16000; // 后端期望的采样率
// AUDIO_CHUNK_SIZE 在 AudioWorkletNode 中定义，这里不再需要
const WS_SEND_INTERVAL_MS = 200; // 每200毫秒发送一次音频数据
let audioQueue = []; // 音频数据队列
let sendIntervalId = null; // 发送定时器ID
let meetingId = `meeting-${Date.now()}`; // 动态生成会议ID
let clientId = `client-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`; // 动态生成客户端ID

// 参与者数据缓存 {userId: {username, role, lastActive, speechCount}}
const participants = {};

// 权限映射 (用于前端显示/隐藏管理面板)
const ROLE_PERMISSIONS = {
    "admin": ["edit_roles", "export_minutes", "export_user_speech", "generate_reports"],
    "manager": ["export_minutes", "export_user_speech", "generate_reports"],
    "team_lead": ["export_minutes", "generate_reports"],
    "member": ["export_minutes"],
    "guest": []
};

let currentUserRole = "guest"; // 假设初始角色为访客，实际应从后端获取或登录后设置

// --- WebSocket 连接管理 ---
function connectWebSocket() {
    if (websocket && (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING)) {
        console.warn("WebSocket已连接或正在连接。");
        return;
    }

    const wsUrl = `${WS_BASE_URL}/${meetingId}/${clientId}`;
    websocket = new WebSocket(wsUrl);

    websocket.onopen = (event) => {
        console.log('WebSocket 连接成功', event);
        wsStatusSpan.textContent = '连接状态: 已连接';
        meetingStatusSpan.textContent = '已连接';
        meetingStatusSpan.classList.remove('bg-secondary', 'bg-danger');
        meetingStatusSpan.classList.add('bg-success');
        // 告知服务器客户端已准备好
        sendMessage({ type: 'client_ready' });
        // 尝试加载初始用户数据到权限管理面板
        loadUsersForAdminPanel();
    };

    websocket.onmessage = (event) => {
        // 关键修正：区分文本消息和二进制消息
        if (event.data instanceof Blob) {
            // 这是音频数据，通常不需要在前端直接处理，除非有播放需求
            // console.log('收到二进制数据 (Blob)，大小:', event.data.size);
        } else if (typeof event.data === 'string') {
            try {
                const message = JSON.parse(event.data);
                console.log('收到 WebSocket 消息:', message); // 调试用

                switch (message.type) {
                    case 'meeting_init_response':
                        console.log(`会议初始化响应: 会议ID=${message.meetingId}, 客户端ID=${message.clientId}, 用户ID=${message.userId}, 角色=${message.role}`);
                        // 设置全局用户ID和角色
                        // HTML中没有display-user-id和display-user-role，所以不更新这些DOM元素
                        currentUserRole = message.role.toLowerCase(); // 更新当前用户角色
                        // 根据用户角色显示或隐藏管理面板和导出按钮
                        checkAdminPanelPermissions();
                        break;
                    case 'realtime_transcript':
                        updateTranscript(message.speakerId, message.userId, message.role, message.text); // 使用speakerId和userId
                        updateCurrentSpeaker(message.speakerId, 100); // 假设实时转录信心度为100%
                        updateParticipantList(message.userId, message.speakerId, message.role);
                        break;
                    case 'realtime_summary':
                        updateLiveSummary(message.summary);
                        break;
                    case 'meeting_topics':
                        // 暂未在HTML中添加话题显示区域，如果需要，请添加
                        console.log('收到会议话题:', message.topics);
                        break;
                    case 'user_role_updated':
                        console.log(`用户 ${message.userId} 的角色已更新为 ${message.newRole}`);
                        updateParticipantRoleInList(message.userId, message.newRole);
                        showToast('角色更新', message.message, 'success');
                        break;
                    case 'error':
                        console.error('WebSocket 错误:', message.message);
                        showToast('错误', message.message, 'danger');
                        break;
                    default:
                        console.warn('收到未知消息类型:', message.type, message);
                }
            } catch (error) {
                console.error('解析 WebSocket 消息失败:', error, event.data);
            }
        } else {
            console.warn('收到未知类型的 WebSocket 数据:', event.data);
        }
    };

    websocket.onerror = (event) => {
        console.error('WebSocket 错误:', event);
        wsStatusSpan.textContent = '连接状态: 错误';
        meetingStatusSpan.textContent = '连接错误';
        meetingStatusSpan.classList.remove('bg-success', 'bg-secondary');
        meetingStatusSpan.classList.add('bg-danger');
        showToast('连接错误', 'WebSocket连接发生错误，请检查网络或后端服务。', 'danger');
    };

    websocket.onclose = (event) => {
        console.log('WebSocket 连接关闭', event);
        wsStatusSpan.textContent = '连接状态: 断开';
        meetingStatusSpan.textContent = '已断开';
        meetingStatusSpan.classList.remove('bg-success', 'bg-danger');
        meetingStatusSpan.classList.add('bg-secondary');
        stopRecording(); // 确保录音停止
        showToast('连接关闭', 'WebSocket连接已关闭。', 'info');
    };
}

function sendMessage(message) {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(message));
        wsQueueSpan.textContent = `待发消息: ${audioQueue.length}`;
    } else {
        console.warn("WebSocket未连接或未打开，无法发送消息:", message);
    }
}

// --- 音频处理和录音控制 ---
async function startRecording() {
    if (isRecording) {
        console.warn("已在录音中。");
        return;
    }

    try {
        // 请求麦克风权限
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        permissionWarning.classList.add('d-none'); // 隐藏权限警告

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        // AudioWorklet加载逻辑已在DOMContentLoaded中调用，这里不再重复
        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');

        audioWorkletNode.port.onmessage = (event) => {
            if (event.data.type === 'audioData') {
                audioQueue.push(event.data.audioData);
                wsQueueSpan.textContent = `待发消息: ${audioQueue.length}`;
            } else if (event.data.type === 'volume') {
                updateVolumeMeter(event.data.volume);
            }
        };

        audioInput = audioContext.createMediaStreamSource(mediaStream);
        audioInput.connect(audioWorkletNode);
        audioWorkletNode.connect(audioContext.destination); // 连接到目的地以保持AudioWorklet运行

        // 启动发送定时器
        sendIntervalId = setInterval(sendAudioChunk, WS_SEND_INTERVAL_MS);

        isRecording = true;
        recordBtn.classList.remove('btn-danger');
        recordBtn.classList.add('btn-success');
        recordBtn.innerHTML = '<i class="bi bi-stop-fill"></i>';
        meetingStatusSpan.textContent = '录音中';
        meetingStatusSpan.classList.remove('bg-secondary', 'bg-danger');
        meetingStatusSpan.classList.add('bg-success');
        transcriptContainer.innerHTML = '<div class="empty-state"><i class="bi bi-mic"></i><p>会议进行中...</p></div>'; // 清空转录区
        emptyStateDiv.classList.add('d-none'); // 隐藏空状态提示
        liveSummaryDiv.innerHTML = '<p class="text-muted">摘要将在会议进行中自动生成...</p>'; // 清空摘要区

        sendMessage({ type: 'start_recording' });
        showToast('录音开始', '麦克风已连接，正在录音...', 'success');

    } catch (error) {
        console.error('获取麦克风失败或AudioWorklet加载失败:', error);
        permissionWarning.classList.remove('d-none'); // 显示权限警告
        recordBtn.classList.remove('btn-success');
        recordBtn.classList.add('btn-danger');
        recordBtn.innerHTML = '<i class="bi bi-mic"></i>';
        meetingStatusSpan.textContent = '麦克风错误';
        meetingStatusSpan.classList.remove('bg-success', 'bg-secondary');
        meetingStatusSpan.classList.add('bg-danger');
        showToast('麦克风错误', '无法访问麦克风。请检查权限设置。', 'danger');
    }
}

function stopRecording() {
    if (!isRecording) {
        console.warn("未在录音中。");
        return;
    }

    isRecording = false;
    recordBtn.classList.remove('btn-success');
    recordBtn.classList.add('btn-danger');
    recordBtn.innerHTML = '<i class="bi bi-mic"></i>';
    meetingStatusSpan.textContent = '已停止';
    meetingStatusSpan.classList.remove('bg-success', 'bg-danger');
    meetingStatusSpan.classList.add('bg-secondary');

    if (sendIntervalId) {
        clearInterval(sendIntervalId);
        sendIntervalId = null;
    }

    if (audioWorkletNode) {
        audioWorkletNode.port.onmessage = null; // 清除消息监听器
        audioWorkletNode.disconnect();
        audioWorkletNode = null;
    }
    if (audioInput) {
        audioInput.disconnect();
        audioInput = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    
    audioQueue = []; // 清空队列
    wsQueueSpan.textContent = `待发消息: 0`;
    resetVolumeMeter(); // 重置音量显示

    sendMessage({ type: 'stop_recording' });
    showToast('录音停止', '录音已停止。', 'info');
}

function sendAudioChunk() {
    if (audioQueue.length > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
        const chunk = audioQueue.shift();
        websocket.send(chunk); // 直接发送 ArrayBuffer
        wsQueueSpan.textContent = `待发消息: ${audioQueue.length}`;
    } else if (audioQueue.length === 0 && isRecording) {
        // console.log("音频队列为空，等待新数据...");
    } else if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        console.warn("WebSocket未打开，停止发送音频。");
        stopRecording();
    }
}

// --- UI 更新函数 ---
function updateTranscript(speakerName, userId, role, text) {
    const speechEntry = speechTemplate.content.cloneNode(true);
    const speakerNameSpan = speechEntry.querySelector('.speaker-name');
    const speechContentDiv = speechEntry.querySelector('.speech-content');
    const speakerTimeSpan = speechEntry.querySelector('.speaker-time');

    speakerNameSpan.textContent = `${speakerName} (${getRoleDisplayName(role)})`;
    speechContentDiv.textContent = text;
    speakerTimeSpan.textContent = new Date().toLocaleTimeString();

    // 根据说话人ID添加特定样式或颜色
    const bubble = speechEntry.querySelector('.speech-bubble');
    // 使用一个更稳定的哈希函数来分配颜色，确保同一个用户ID总是获得相同的颜色
    const hashCode = userId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const colorIndex = hashCode % 5; // 假设有5种颜色类
    bubble.classList.add(`speaker-color-${colorIndex}`); 

    transcriptContainer.appendChild(speechEntry);
    transcriptContainer.scrollTop = transcriptContainer.scrollHeight; // 滚动到底部
}

function updateCurrentSpeaker(speakerName, confidence) {
    currentSpeakerSpan.textContent = `当前发言人: ${speakerName}`;
    speakerConfidenceSpan.textContent = `${confidence}%`;
}

function updateLiveSummary(summaryText) {
    liveSummaryDiv.innerHTML = `<p>${marked.parse(summaryText)}</p>`; // 使用marked.js解析Markdown
    liveSummaryDiv.scrollTop = liveSummaryDiv.scrollHeight;
}

let volumePeakValue = 0;
let volumeHistory = []; // 用于计算平均音量

function updateVolumeMeter(volume) {
    const percentage = Math.min(100, Math.round(volume * 100)); // 将0-1范围转换为0-100%

    volumeMeter.style.width = `${percentage}%`;
    volumeMeter.setAttribute('aria-valuenow', percentage);
    volumeText.textContent = `${percentage}%`;

    // 更新峰值
    if (percentage > volumePeakValue) {
        volumePeakValue = percentage;
        volumePeak.textContent = `${volumePeakValue}%`;
    }

    // 更新平均值 (简单移动平均)
    volumeHistory.push(percentage);
    if (volumeHistory.length > 50) { // 保持最近50个数据点
        volumeHistory.shift();
    }
    const avgVolume = volumeHistory.reduce((a, b) => a + b, 0) / volumeHistory.length;
    volumeAvg.textContent = `${Math.round(avgVolume)}%`;

    // 根据音量调整进度条颜色
    if (percentage > 80) {
        volumeMeter.className = 'progress-bar volume-indicator bg-danger';
    } else if (percentage > 50) {
        volumeMeter.className = 'progress-bar volume-indicator bg-warning';
    } else {
        volumeMeter.className = 'progress-bar volume-indicator bg-success';
    }
}

function resetVolumeMeter() {
    volumePeakValue = 0;
    volumeHistory = [];
    volumeMeter.style.width = '0%';
    volumeMeter.setAttribute('aria-valuenow', '0');
    volumeText.textContent = '0% (未激活)';
    volumePeak.textContent = '0%';
    volumeAvg.textContent = '0%';
    volumeMeter.className = 'progress-bar volume-indicator'; // 重置颜色
}

function updateParticipantList(userId, username, role) {
    if (!participants[userId]) {
        participants[userId] = { username: username, role: role, lastActive: new Date(), speechCount: 0 };
    }
    participants[userId].lastActive = new Date(); // 更新活跃时间
    participants[userId].speechCount++; // 每次发言增加计数
    participants[userId].username = username; // 更新用户名，以防“未知说话人”被识别
    participants[userId].role = role; // 更新角色，以防识别后角色更新

    renderParticipantList();
}

function renderParticipantList() {
    participantsList.innerHTML = ''; // 清空列表

    const sortedParticipants = Object.values(participants).sort((a, b) => {
        // 优先按角色排序 (管理员 > 经理 > 团队领导 > 成员 > 访客)
        const roleOrder = { 'admin': 0, 'manager': 1, 'team_lead': 2, 'member': 3, 'guest': 4 };
        const roleComparison = (roleOrder[a.role.toLowerCase()] || 99) - (roleOrder[b.role.toLowerCase()] || 99); // 处理未知角色
        if (roleComparison !== 0) return roleComparison;
        // 然后按发言次数倒序
        return b.speechCount - a.speechCount;
    });

    sortedParticipants.forEach(p => {
        const participantEntry = participantTemplate.content.cloneNode(true);
        const link = participantEntry.querySelector('a');
        link.dataset.userId = p.userId; // 存储用户ID
        link.dataset.username = p.username;
        link.dataset.role = p.role;

        participantEntry.querySelector('.participant-name').textContent = p.username;
        participantEntry.querySelector('.participant-role').textContent = `角色: ${getRoleDisplayName(p.role)}`;
        participantEntry.querySelector('.last-active').textContent = `最后活跃: ${p.lastActive.toLocaleTimeString()}`;
        participantEntry.querySelector('.speech-count').textContent = `发言: ${p.speechCount}次`;
        participantsList.appendChild(participantEntry);
    });

    updateParticipantCount();
}


function updateParticipantRoleInList(userId, newRole) {
    if (participants[userId]) {
        participants[userId].role = newRole;
        renderParticipantList(); // 重新渲染列表以反映角色变化
        // 更新用户选择下拉菜单中的文本
        const optionToUpdate = userSelect.querySelector(`option[value="${userId}"]`);
        if (optionToUpdate) {
            optionToUpdate.textContent = `${participants[userId].username} (${getRoleDisplayName(newRole)})`;
        }
    }
    // 重新评估权限面板的可见性
    checkAdminPanelPermissions();
}

function updateParticipantCount() {
    document.getElementById('participant-count').textContent = `${Object.keys(participants).length}人`;
}

function getRoleDisplayName(role) {
    switch (role.toLowerCase()) { // 统一转换为小写进行比较
        case 'admin': return '管理员';
        case 'manager': return '经理';
        case 'team_lead': return '团队领导';
        case 'member': return '成员';
        case 'guest': return '访客';
        default: return role;
    }
}

// --- 权限管理面板逻辑 ---
function checkAdminPanelPermissions() {
    // 检查权限并显示/隐藏面板
    const userPermissions = ROLE_PERMISSIONS[currentUserRole] || [];

    // 权限管理面板
    const adminPanelElement = document.querySelector('.admin-panel');
    if (adminPanelElement) {
        if (userPermissions.includes("edit_roles")) {
            adminPanelElement.style.display = 'block';
        } else {
            adminPanelElement.style.display = 'none';
        }
    }

    // 导出面板及其按钮
    const exportPanelElement = document.querySelector('.export-panel');
    let anyExportButtonVisible = false;
    if (exportPanelElement) {
        document.querySelectorAll('[data-permission]').forEach(element => {
            const requiredPermission = element.dataset.permission;
            if (userPermissions.includes(requiredPermission)) {
                element.style.display = 'block'; // 或 'flex', 'inline-block' 根据元素类型
                anyExportButtonVisible = true;
            } else {
                element.style.display = 'none';
            }
        });

        if (anyExportButtonVisible) {
            exportPanelElement.style.display = 'block';
        } else {
            exportPanelElement.style.display = 'none';
        }
    }
}

async function loadUsersForAdminPanel() {
    userSelect.innerHTML = '<option value="">-- 加载用户中... --</option>';
    try {
        const response = await fetch(`${API_BASE_URL}/get-all-users`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const users = await response.json();
        userSelect.innerHTML = '<option value="">-- 请选择用户 --</option>'; // 清空并添加默认选项
        if (users.length === 0) {
            userSelect.innerHTML = '<option value="">暂无注册用户。</option>';
        } else {
            users.forEach(user => {
                const option = document.createElement('option');
                option.value = user.user_id;
                option.textContent = `${user.username} (ID: ${user.user_id.substring(0, 8)}... | 角色: ${getRoleDisplayName(user.role)})`;
                userSelect.appendChild(option);
                // 同时更新本地缓存，用于参会人员列表
                participants[user.user_id] = { 
                    username: user.username, 
                    role: user.role, 
                    lastActive: user.last_active ? new Date(user.last_active) : new Date(), // Convert ISO string to Date
                    speechCount: 0 // 假设初始为0，或从后端获取
                };
            });
            renderParticipantList(); // 重新渲染参会人员列表
        }
    } catch (error) {
        console.error("加载用户列表失败:", error);
        showToast('加载用户失败', `无法获取用户列表: ${error.message}`, 'danger');
    }
}

async function updateRole() {
    const selectedUserId = userSelect.value;
    const newRole = roleSelect.value;
    const updateRoleStatus = document.getElementById('update-role-status'); // 假设HTML中有这个元素

    if (!selectedUserId) {
        showToast('错误', '请选择一个用户。', 'warning');
        return;
    }

    if (!newRole) {
        showToast('错误', '请选择一个新角色。', 'warning');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/update-role?voiceprint_id=${selectedUserId}&new_role=${newRole}`, {
            method: 'POST',
            headers: {
                'Accept': 'application/json'
                // 'Authorization': 'Bearer YOUR_AUTH_TOKEN' // 如果需要认证
            }
        });

        const result = await response.json();
        if (response.ok) {
            showToast('成功', result.message, 'success');
            updateParticipantRoleInList(selectedUserId, newRole); // 更新前端显示
            loadUsersForAdminPanel(); // 刷新用户选择下拉菜单
        } else {
            throw new Error(result.detail || result.message || '更新角色失败');
        }
    } catch (error) {
        console.error("更新角色失败:", error);
        showToast('更新失败', error.message, 'danger');
    }
}

// --- 导出功能 ---
async function exportMinutes() {
    if (!meetingId || meetingId.startsWith('meeting-')) {
        showToast('错误', '请先进行一次会议录音，以获取有效的会议ID。', 'warning');
        return;
    }
    try {
        showToast('正在生成会议纪要，请稍候...', 'info');
        // 后端 /generate_minutes 返回的是一个 DOCX 文件流
        const response = await fetch(`${API_BASE_URL}/generate_minutes?meeting_id=${meetingId}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `会议纪要-${meetingId}.docx`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('导出成功', '会议纪要已生成并下载。', 'success');
    } catch (error) {
        console.error("导出会议纪要失败:", error);
        showToast('导出失败', error.message, 'danger');
    }
}

async function exportUserSpeeches() {
    if (!meetingId || meetingId.startsWith('meeting-')) {
        showToast('错误', '请先进行一次会议录音，以获取有效的会议ID。', 'warning');
        return;
    }
    const userIdToExport = prompt("请输入要导出的用户ID (留空则导出所有用户发言):");
    let url = `${API_BASE_URL}/export_user_speech?meeting_id=${meetingId}`;
    if (userIdToExport) {
        url += `&user_id=${encodeURIComponent(userIdToExport)}`;
    }

    try {
        showToast('正在导出用户发言，请稍候...', 'info');
        const response = await fetch(url);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        const blob = await response.blob();
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = downloadUrl;
        a.download = `用户发言-${meetingId}${userIdToExport ? '-' + userIdToExport.substring(0,8) : ''}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(downloadUrl);
        showToast('导出成功', '用户发言已成功导出！', 'success');
    } catch (error) {
        console.error('导出用户发言失败:', error);
        showToast('导出失败', `导出用户发言失败: ${error.message}`, 'danger');
    }
}


async function generateReport() {
    if (!meetingId || meetingId.startsWith('meeting-')) {
        showToast('错误', '请先进行一次会议录音，以获取有效的会议ID。', 'warning');
        return;
    }
    try {
        showToast('正在生成总结报告，请稍候...', 'info');
        // 后端 /generate_meeting_report 返回的是 JSON 格式的报告内容
        const response = await fetch(`${API_BASE_URL}/generate_meeting_report?meeting_id=${meetingId}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        const data = await response.json();
        if (data.report && data.report.content) {
            // 将Markdown内容转换为Blob并提供下载，或者在模态框中显示
            const reportContent = data.report.content;
            const blob = new Blob([reportContent], { type: 'text/markdown;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `会议报告-${meetingId}.md`; // 下载为Markdown文件
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showToast('报告生成成功', '会议报告已生成并下载为Markdown文件。', 'success');
        } else {
            showToast('报告生成失败', '生成的报告内容为空。', 'warning');
        }
    } catch (error) {
        console.error("生成总结报告失败:", error);
        showToast('生成失败', error.message, 'danger');
    }
}

// --- 消息提示 (Toast) ---
function showToast(title, message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || (() => {
        const div = document.createElement('div');
        div.id = 'toast-container';
        div.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(div);
        return div;
    })();

    const toastHtml = `
        <div class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${title}:</strong> ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    const div = document.createElement('div');
    div.innerHTML = toastHtml;
    const toastElement = div.firstElementChild;
    toastContainer.appendChild(toastElement);

    const toast = new bootstrap.Toast(toastElement);
    toast.show();

    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}


// --- AudioWorklet加载逻辑 ---
async function loadAudioProcessor() {
    try {
        // 确保 audioContext 存在，如果不存在则创建
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        }
        await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');
        console.log("AudioWorklet处理器加载成功");
    } catch (e) {
        console.error("AudioWorklet加载失败，将使用兼容模式", e);
        // 如果AudioWorklet加载失败，可以考虑回退到ScriptProcessorNode
        // 但为了简化，目前只记录错误
    }
}


// --- 事件监听器 ---
recordBtn.addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
    } else {
        connectWebSocket(); // 确保WebSocket已连接
        startRecording();
    }
});

updateRoleBtn.addEventListener('click', updateRole);
exportMinutesBtn.addEventListener('click', exportMinutes);
exportSpeechesBtn.addEventListener('click', exportUserSpeeches); // 绑定导出用户发言按钮
generateReportBtn.addEventListener('click', generateReport);


// --- 初始化 ---
document.addEventListener('DOMContentLoaded', async () => {
    // 检查麦克风权限状态
    try {
        const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
        if (permissionStatus.state === 'granted') {
            permissionWarning.classList.add('d-none');
        } else {
            permissionWarning.classList.remove('d-none');
        }
        permissionStatus.onchange = () => {
            if (permissionStatus.state === 'granted') {
                permissionWarning.classList.add('d-none');
            } else {
                permissionWarning.classList.remove('d-none');
                stopRecording(); // 如果权限被撤销，停止录音
            }
        };
    } catch (e) {
        console.warn("浏览器不支持 Permissions API 或查询麦克风权限失败", e);
        // 如果不支持，就保持警告可见，直到用户尝试录音
    }

    // 初始隐藏管理面板和导出面板，直到用户角色确定
    adminPanel.style.display = 'none';
    const exportPanelElement = document.querySelector('.export-panel');
    if (exportPanelElement) {
        exportPanelElement.style.display = 'none';
    }

    // 确保 AudioWorklet 处理器在页面加载时被加载
    await loadAudioProcessor();

    // 初始加载用户列表到管理面板 (在AudioWorklet加载后，确保所有初始化完成)
    loadUsersForAdminPanel();
    
    // 初始设置按钮状态和显示
    recordBtn.classList.remove('recording');
    meetingStatusSpan.textContent = '未开始';
    resetVolumeMeter(); // 确保音量表初始状态正确
});


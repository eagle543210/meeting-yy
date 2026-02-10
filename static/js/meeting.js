class MeetingUI {
    constructor() {
        // 1. 初始化所有属性
        this.speakers = {};
        this.currentSpeaker = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.recording = false;
        this.speechHistory = [];
        this.speakerColors = [
            '#0d6efd', '#198754', '#dc3545', '#ffc107', 
            '#0dcaf0', '#6610f2', '#fd7e14', '#20c997'
        ];
        this.userPermissions = [];
        this.meetingId = this._generateMeetingId();
        this.voiceprintId = localStorage.getItem('voiceprint_id') || null;
        this.audioInterval = null;
        this.volumeMeter = null;
        this.volumeText = null;
        this.volumePeak = null;
        this.volumeAvg = null;
        this.volumeHistory = [];
        this.analyser = null;
        this.volumeInterval = null;
        this.dataArray = null;
        this.errorCount = 0;
        this.maxErrorCount = 3;
        this.emptyStateEl = document.querySelector('.empty-state');
        this.heartbeatInterval = null;
        this.lastHeartbeatResponse = null;
        this.connectionTimeout = null;
        this.audioSendInterval = 200; // 发送间隔(ms)
        this.lastAudioSendTime = 0;
        this.isProcessingAudio = false;
        // 2. 显式绑定所有方法
        this.initElements = this._initElements.bind(this);
        this.initEventListeners = this._initEventListeners.bind(this);
        this.initMeeting = this._initMeeting.bind(this);
        this.toggleRecording = this._toggleRecording.bind(this);
        this.startRecording = this._startRecording.bind(this);
        this.stopRecording = this._stopRecording.bind(this);
        this.handleServerMessage = this._handleServerMessage.bind(this);
        this.showError = this._showError.bind(this);
        this.updateRecordingUI = this._updateRecordingUI.bind(this);
        this.addSpeechBubble = this._addSpeechBubble.bind(this);
        this.updateSummary = this._updateSummary.bind(this);
        this.handleError = this._handleError.bind(this);
        this.setupAudioProcessing = this._setupAudioProcessing.bind(this);
        this.setupScriptProcessor = this._setupScriptProcessor.bind(this);
        this.startVolumeMonitoring = this._startVolumeMonitoring.bind(this);
        this.updateVolumeMeter = this._updateVolumeMeter.bind(this);
        this.addParticipant = this._addParticipant.bind(this);

        // 3. 初始化UI组件
        this._initElements();
        this._initVolumeElements();
        this._initEventListeners();
        
        // 4. 初始化会议连接
        this._initMeeting();
    }

    _generateMeetingId() {
        const timestamp = Date.now().toString(36);
        const randomStr = Math.random().toString(36).substr(2, 5);
        return `meeting-${timestamp}-${randomStr}`;
    }

    _initElements() {
        this.recordBtn = document.getElementById('record-btn');
        this.transcriptContainer = document.getElementById('transcript-container');
        this.currentSpeakerEl = document.getElementById('current-speaker');
        this.speakerConfidenceEl = document.getElementById('speaker-confidence');
        this.meetingStatusEl = document.getElementById('meeting-status');
        this.participantsList = document.getElementById('participants-list');
        this.liveSummaryEl = document.getElementById('live-summary');
        this.speechTemplate = document.getElementById('speech-template');
        this.participantTemplate = document.getElementById('participant-template');
    }

    _initVolumeElements() {
        this.volumeMeter = document.getElementById('volume-meter');
        this.volumeText = document.getElementById('volume-text');
        this.volumePeak = document.getElementById('volume-peak');
        this.volumeAvg = document.getElementById('volume-avg');
    }

    _initEventListeners() {
        this.recordBtn.addEventListener('click', () => this._toggleRecording());
    }

    _initMeeting() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.meetingId}`;
            
            this.wsManager = new WebSocketManager(wsUrl);
            
            this.wsManager.on('connected', () => {
                console.log("WebSocket连接已建立");
                this.wsManager.send({
                    type: 'meeting_init',  // 修改消息类型为更明确的meeting_init
                    meetingId: this.meetingId,
                    voiceprintId: this.voiceprintId
                });
            });
            
            this.wsManager.on('message', (data) => {
                this._handleServerMessage(data);
            });
            
            this.wsManager.on('error', (error) => {
                console.error("WebSocket错误:", error);
                this._showError(`连接错误: ${error.message}`);
            });
            
        } catch (error) {
            console.error("会议初始化失败:", error);
            this._showError("连接服务器失败，请刷新页面重试");
        }
    }


    // 在MeetingUI类中添加以下方法
_updateSpeakerInfo(speakerId, confidence) {
    this.currentSpeakerEl.textContent = speakerId || "等待识别发言人...";
    
    if (confidence) {
        this.speakerConfidenceEl.textContent = `${Math.round(confidence * 100)}%`;
        this.speakerConfidenceEl.className = `badge ${
            confidence > 0.8 ? 'bg-success' : 
            confidence > 0.5 ? 'bg-warning' : 'bg-danger'
        }`;
    }
}

_addParticipant(userId, role = "成员") {
    if (!this.speakers[userId]) {
        const colorIndex = Object.keys(this.speakers).length % this.speakerColors.length;
        this.speakers[userId] = {
            id: userId,
            role: role,
            color: this.speakerColors[colorIndex],
            speechCount: 0,
            lastActive: new Date()
        };
        
        const participantEl = document.createElement('a');
        participantEl.className = 'list-group-item list-group-item-action';
        participantEl.innerHTML = `
            <div class="d-flex w-100 justify-content-between">
                <h6 class="mb-1" style="color: ${this.speakerColors[colorIndex]}">${userId}</h6>
                <small class="text-muted">刚刚加入</small>
            </div>
            <p class="mb-1 small">角色: ${role}</p>
            <small class="text-muted">发言: 0次</small>
        `;
        
        this.participantsList.appendChild(participantEl);
        document.getElementById('participant-count').textContent = `${Object.keys(this.speakers).length}人`;
    }
}
    async _toggleRecording() {
        try {
            if (this.recording) {
                this._stopRecording();
            } else {
                await this._startRecording();
            }
        } catch (error) {
            console.error("切换录音状态出错:", error);
            this._showError("无法切换录音状态: " + error.message);
        }
    }

    async _startRecording() {
    if (this.recording) return;
    
    try {
        // 重置状态
        this.lastAudioSendTime = 0;
        this.isProcessingAudio = false;
        
        // 初始化音频
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // 设置处理
        await this._setupAudioProcessing();
        
        this.recording = true;
        this._updateRecordingUI(true);
        
        this.wsManager.send({
            type: 'recording_control',
            action: 'start',
            meetingId: this.meetingId,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error("录音启动失败:", error);
        this._stopRecording();
        this._showError("录音启动失败: " + error.message);
    }
}

    _stopRecording() {
    if (!this.recording) return;
    
    this.recording = false;
    this._updateRecordingUI(false);
    
    // 清理资源
    if (this.mediaStream) {
        this.mediaStream.getTracks().forEach(track => track.stop());
        this.mediaStream = null;
    }
    
    if (this.audioContext) {
        this.audioContext.close().catch(console.error);
        this.audioContext = null;
    }
    
    this.wsManager.send({
        type: 'recording_control',
        action: 'stop',
        meetingId: this.meetingId,
        timestamp: new Date().toISOString()
    });
}

   
    
_handleServerMessage(data) {
    console.groupCollapsed(`处理服务器消息 [${data.type}]`);
    console.log("完整消息:", data);
    
    try {
        switch(data.type) {
            case 'speaker_identified':
                console.log("更新发言人信息");
                this._updateSpeakerInfo(data.speaker_id, data.confidence);
                this._addParticipant(data.speaker_id, data.role);
                break;
                
            case 'transcript_result':
                console.log("显示识别结果");
                this._addSpeechBubble(
                    data.speaker_id || "未知发言人", 
                    data.transcript,
                    data.confidence
                );
                break;
                
            case 'recording_status':
                console.log("更新录音状态:", data.status);
                this._updateRecordingUI(data.status === 'recording_started');
                break;
                
            case 'error':
                console.error("服务器错误:", data.message);
                this._showError(data.message);
                break;
                
            default:
                console.warn("未处理的消息类型:", data.type);
        }
    } catch (error) {
        console.error("消息处理出错:", error);
    } finally {
        console.groupEnd();
    }
}

_updateRecordingStatus(status) {
    const statusMap = {
        'recording_started': { text: '录音中', class: 'bg-success' },
        'recording_stopped': { text: '已停止', class: 'bg-secondary' }
    };
    
    if (statusMap[status]) {
        this.meetingStatusEl.textContent = statusMap[status].text;
        this.meetingStatusEl.className = `badge ${statusMap[status].class}`;
    }
}

    _handleMeetingUpdate(data) {
        if (data.speaker) {
            this.currentSpeaker = data.speaker;
            this.currentSpeakerEl.textContent = data.speaker;
            if (data.confidence) {
                this.speakerConfidenceEl.textContent = `${Math.round(data.confidence * 100)}%`;
                this.speakerConfidenceEl.className = `badge ${
                    data.confidence > 0.8 ? 'bg-success' : 
                    data.confidence > 0.5 ? 'bg-warning' : 'bg-danger'
                }`;
            }
        }
        
        if (data.transcript) {
            this._addSpeechBubble(data.speaker || "未知发言人", data.transcript);
        }
        
        if (data.summary) {
            this._updateSummary(data.summary);
        }
    }

    _addSpeechBubble(speakerId, text, confidence = 0) {
    // 确保空状态隐藏
    const emptyState = this.transcriptContainer.querySelector('.empty-state');
    if (emptyState) emptyState.style.display = 'none';
    
    // 创建气泡元素
    const bubble = document.createElement('div');
    bubble.className = 'speech-bubble';
    bubble.innerHTML = `
        <div class="speaker-header">
            <span class="speaker-name" style="color: ${this._getSpeakerColor(speakerId)}">
                ${speakerId}
                <span class="confidence-badge">${Math.round(confidence * 100)}%</span>
            </span>
            <span class="speaker-time">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="speech-content">${text}</div>
    `;
    
    // 添加到容器
    this.transcriptContainer.appendChild(bubble);
    this.transcriptContainer.scrollTop = this.transcriptContainer.scrollHeight;
    
    console.log("已添加发言气泡:", {speakerId, text});
}

_getSpeakerColor(speakerId) {
    if (!this.speakers[speakerId]) {
        const colors = ['#0d6efd', '#198754', '#dc3545', '#fd7e14'];
        this.speakers[speakerId] = {
            color: colors[Object.keys(this.speakers).length % colors.length]
        };
    }
    return this.speakers[speakerId].color;
}
    _updateSummary(summary) {
        this.liveSummaryEl.innerHTML = summary;
    }

    _updateParticipantSpeechCount(speakerId) {
    const participantItems = this.participantsList.querySelectorAll('.list-group-item');
    participantItems.forEach(item => {
        if (item.querySelector('.participant-name').textContent === speakerId) {
            const countEl = item.querySelector('.speech-count');
            const currentCount = parseInt(countEl.textContent.match(/\d+/)[0]) || 0;
            countEl.textContent = `发言: ${currentCount + 1}次`;
        }
    });
}

    _handleError(data) {
        console.error("服务器错误:", data.message);
        this._showError(data.message || "服务器发生错误");
        
        if (data.message.includes('audio') || data.message.includes('处理')) {
            this._stopRecording();
        }
    }

    async _transcribeAudio(audioData) {
        try {
            // 这里是模拟实现，实际项目应该调用ASR服务
            const mockTranscripts = [
                "我们今天讨论项目进度",
                "需要加快开发速度",
                "下周进行代码评审",
                "用户反馈需要优先处理"
            ];
            
            // 模拟处理延迟
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // 返回随机模拟文本
            return mockTranscripts[Math.floor(Math.random() * mockTranscripts.length)];
        } catch (error) {
            console.error("语音转文字失败:", error);
            return "[语音识别失败]";
        }
    }

   
// MeetingUI类
async _setupAudioProcessing() {
    try {
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.volumeDataArray = new Uint8Array(this.analyser.frequencyBinCount);
        
        source.connect(this.analyser);
        
        // 使用ScriptProcessor控制发送频率
        const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (e) => {
            if (!this.recording || this.isProcessingAudio) return;
            
            const now = Date.now();
            if (now - this.lastAudioSendTime < this.audioSendInterval) return;
            
            this.isProcessingAudio = true;
            try {
                const inputData = e.inputBuffer.getChannelData(0);
                const audioData = new Float32Array(inputData.length);
                
                // 计算音量
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) {
                    audioData[i] = inputData[i];
                    sum += inputData[i] * inputData[i];
                }
                const rms = Math.sqrt(sum / inputData.length);
                const volume = Math.min(100, Math.round(rms * 1000));
                this._updateVolumeMeter(volume);
                
                // 控制发送频率
                if (now - this.lastAudioSendTime >= this.audioSendInterval) {
                    this.wsManager.send({
                        type: 'audio_chunk',
                        meetingId: this.meetingId,
                        voiceprintId: this.voiceprintId || 'unknown',
                        audioData: Array.from(audioData),
                        sampleRate: this.audioContext.sampleRate,
                        timestamp: new Date().toISOString()
                    });
                    this.lastAudioSendTime = now;
                }
            } finally {
                this.isProcessingAudio = false;
            }
        };
        
        source.connect(processor);
        processor.connect(this.audioContext.destination);
        
    } catch (error) {
        console.error("音频处理设置失败:", error);
        throw error;
    }
}

    _setupScriptProcessor(source) {
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        this.audioBufferQueue = [];
        
        source.connect(this.analyser);
        this.analyser.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        
        this.processor.onaudioprocess = (e) => {
            if (!this.recording) return;
            
            const audioData = e.inputBuffer.getChannelData(0);
            this.audioBufferQueue.push(audioData.slice(0));
            
            this.analyser.getByteFrequencyData(this.volumeDataArray);
            this._updateVolumeMeter(this.volumeDataArray);
        };
        
        this.audioInterval = setInterval(() => {
            if (!this.recording || this.audioBufferQueue.length === 0) return;
            
            const audioData = this.audioBufferQueue.shift();
            this.wsManager.send({
                type: 'audio_chunk',
                voiceprint_id: this.voiceprintId || 'default_user',
                data: Array.from(audioData),
                sampleRate: this.audioContext.sampleRate,
                timestamp: new Date().toISOString()
            });
        }, 100);
    }

    _startVolumeMonitoring() {
    if (this.volumeInterval) clearInterval(this.volumeInterval);
    
    this.volumeInterval = setInterval(() => {
        if (!this.analyser || !this.recording) return;
        
        this.analyser.getByteFrequencyData(this.volumeDataArray);
        
        // 计算平均音量
        let sum = 0;
        for (let i = 0; i < this.volumeDataArray.length; i++) {
            sum += this.volumeDataArray[i];
        }
        const average = sum / this.volumeDataArray.length;
        const volumeLevel = Math.min(100, Math.round(average));
        
        // 更新UI
        this._updateVolumeMeter(volumeLevel);
    }, 100);
}

    _updateVolumeMeter(volumeLevel) {
    if (!this.volumeMeter || !this.volumeText) return;
    
    this.volumeMeter.style.width = `${volumeLevel}%`;
    this.volumeText.textContent = `${volumeLevel}% (${volumeLevel > 50 ? '活跃' : '安静'})`;
    
    // 更新峰值和平均值
    this.volumeHistory.push(volumeLevel);
    if (this.volumeHistory.length > 10) {
        this.volumeHistory.shift();
    }
    
    const peak = Math.max(...this.volumeHistory);
    const avg = Math.round(this.volumeHistory.reduce((a, b) => a + b, 0) / this.volumeHistory.length);
    
    this.volumePeak.textContent = peak;
    this.volumeAvg.textContent = avg;
    
    // 根据音量级别改变颜色
    if (volumeLevel > 70) {
        this.volumeMeter.style.backgroundColor = '#dc3545';
    } else if (volumeLevel > 30) {
        this.volumeMeter.style.backgroundColor = '#ffc107';
    } else {
        this.volumeMeter.style.backgroundColor = '#28a745';
    }
}

    _updateRecordingUI(isRecording) {
        if (!this.recordBtn || !this.meetingStatusEl) return;
        
        this.recordBtn.innerHTML = isRecording 
            ? '<i class="bi bi-stop-fill"></i> 停止录音'
            : '<i class="bi bi-mic"></i> 开始录音';
        
        this.recordBtn.classList.toggle('btn-primary', !isRecording);
        this.recordBtn.classList.toggle('btn-danger', isRecording);
        
        this.meetingStatusEl.textContent = isRecording ? '进行中' : '已结束';
        this.meetingStatusEl.className = `badge ${isRecording ? 'bg-success' : 'bg-secondary'}`;
    }

    _showError(message) {
        const errorEl = document.getElementById('recording-error');
        if (!errorEl) {
            const container = document.querySelector('.card-footer');
            const el = document.createElement('div');
            el.id = 'recording-error';
            el.className = 'alert alert-danger mt-2';
            el.textContent = message;
            container.appendChild(el);
        } else {
            errorEl.textContent = message;
        }
        
        setTimeout(() => {
            const el = document.getElementById('recording-error');
            if (el) el.remove();
        }, 5000);
    }
}

class WebSocketManager {
    constructor(url) {
        this.url = url;
        this.socket = null;
        this.queue = [];
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.eventListeners = {};
        this.pendingMessages = new Map();
        this.messageId = 0;
        this.heartbeatIntervalId = null;
        this.heartbeatTimeoutId = null;
        this.lastHeartbeatResponse = null;

        // 绑定方法
        this.connect = this._connect.bind(this);
        this.disconnect = this._disconnect.bind(this);
        this.send = this._send.bind(this);
        this.startHeartbeat = this._startHeartbeat.bind(this);
        this.stopHeartbeat = this._stopHeartbeat.bind(this);
        this.reconnect = this._reconnect.bind(this);
        this.handleMessage = this._handleMessage.bind(this);
        this.emit = this._emit.bind(this);
        this.on = this._on.bind(this);
        this.isConnected = this._isConnected.bind(this);
        this.flushQueue = this._flushQueue.bind(this);
    }

    async _connect() {
        if (this.isConnecting || this._isConnected()) return;
        
        this.isConnecting = true;
        this.socket = new WebSocket(this.url);

        this.socket.onopen = () => {
            this.isConnecting = false;
            this.reconnectAttempts = 0;
            this._emit('connected');
            this._flushQueue();
            this._startHeartbeat();
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this._handleMessage(data);
            } catch (error) {
                console.error("消息解析错误:", error);
                this._emit('error', new Error('消息解析失败'));
            }
        };

        this.socket.onerror = (error) => {
            console.error("WebSocket错误:", error);
            this.isConnecting = false;
            this._emit('error', error);
        };

        this.socket.onclose = () => {
            console.log("WebSocket连接关闭");
            this.isConnecting = false;
            this._emit('disconnected');
            this._reconnect();
        };
    }

    _disconnect() {
        if (this.socket) {
            this.socket.close();
        }
    }

    _isConnected() {
        return this.socket && this.socket.readyState === WebSocket.OPEN;
    }

    _send(message) {
        if (this._isConnected()) {
            try {
                this.socket.send(JSON.stringify(message));
            } catch (error) {
                console.error("发送消息失败:", error);
                this.queue.push(message);
                this._reconnect();
            }
        } else {
            this.queue.push(message);
            if (!this.isConnecting) {
                this._connect();
            }
        }
    }

    _startHeartbeat() {
        this._stopHeartbeat();
        
        this.heartbeatIntervalId = setInterval(() => {
            const heartbeatMsg = {
                type: 'heartbeat',
                timestamp: Date.now()
            };
            this._send(heartbeatMsg);
            
            this.lastHeartbeatResponse = Date.now();
            this.heartbeatTimeoutId = setTimeout(() => {
                if (Date.now() - this.lastHeartbeatResponse > 5000) {
                    console.warn("心跳响应超时，尝试重连");
                    this._reconnect();
                }
            }, 5000);
        }, 30000);
    }

    _stopHeartbeat() {
        if (this.heartbeatIntervalId) {
            clearInterval(this.heartbeatIntervalId);
            this.heartbeatIntervalId = null;
        }
        if (this.heartbeatTimeoutId) {
            clearTimeout(this.heartbeatTimeoutId);
            this.heartbeatTimeoutId = null;
        }
    }

    _reconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error("达到最大重连次数，停止尝试");
            this._emit('connection_failed');
            return;
        }

        const delay = Math.min(30000, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
        this.reconnectAttempts++;
        
        setTimeout(() => {
            if (!this._isConnected() && !this.isConnecting) {
                this._connect();
            }
        }, delay);
    }

    _handleMessage(data) {
    console.log("收到服务器原始消息:", data);  // 调试原始数据
    
    try {
        // 确保数据是对象
        const message = typeof data === 'string' ? JSON.parse(data) : data;
        console.log("解析后的服务器消息:", message);  // 调试解析后的数据
        
        if (message.messageId && this.pendingMessages.has(message.messageId)) {
            const { resolve, timeoutId } = this.pendingMessages.get(message.messageId);
            clearTimeout(timeoutId);
            this.pendingMessages.delete(message.messageId);
            resolve(message);
            return;
        }
        
        this._emit('message', message);
    } catch (error) {
        console.error("消息解析错误:", error, "原始数据:", data);
        this._emit('error', new Error('消息解析失败'));
    }
}

    _emit(event, ...args) {
        const listeners = this.eventListeners[event];
        if (listeners) {
            listeners.forEach(callback => callback(...args));
        }
    }

    _on(event, callback) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        this.eventListeners[event].push(callback);
    }

    _flushQueue() {
        while (this.queue.length > 0 && this._isConnected()) {
            const message = this.queue.shift();
            this._send(message);
        }
    }
}

// 安全初始化
document.addEventListener('DOMContentLoaded', () => {
    try {
        new MeetingUI();
    } catch (error) {
        console.error("全局错误:", error);
        alert("系统初始化失败: " + (error.message || "未知错误"));
    }
});

// 全局错误处理
window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的Promise拒绝:', event.reason);
});

window.addEventListener('error', (event) => {
    console.error('全局错误:', event.error);
});
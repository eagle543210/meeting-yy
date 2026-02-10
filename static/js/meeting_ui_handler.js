// M:\meeting\frontend/js/meeting_ui_handler.js

import { WebSocketClient } from './websocket_client.js';
import { AudioRecorder } from './audio_recorder.js';

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

export class MeetingUI {
    constructor() {
        this.elements = this._getDOMElements();
        this.isRecording = false;
        this.wsClient = null;
        this.audioRecorder = null;
        
        // 用户管理
        this.knownUsers = new Map(); // client_id -> {user_id, user_name, role}
        this.userCounter = 1;
        this.baseUserName = "用户";
        
        // 转录状态
        this.activeSpeaker = null;
        this.lastSpeakerId = null;
        this.lastAppendedEntry = null;
        this.lastTranscriptContent = "";
        this.finalizationTimeout = 1500;
        this.finalizationCheckInterval = null;
        this.lastReceivedTranscriptTime = 0;

        this._setupEventListeners();
        this._updateMeetingStatus('未开始', 'secondary');
        this._updateConnectionStatus('断开', 0, 'danger');
        this._updateStatus("点击开始录音以启动会议", 'info');
    }

    _getDOMElements() {
        return {
            recordBtn: document.getElementById('record-btn'),
            transcriptContainer: document.getElementById('transcript-container'),
            permissionWarning: document.getElementById('permission-warning'),
            meetingStatus: document.getElementById('meeting-status'),
            currentSpeaker: document.getElementById('current-speaker'),
            speakerConfidence: document.getElementById('speaker-confidence'),
            participantsList: document.getElementById('participants-list'),
            participantCount: document.getElementById('participant-count'),
            userSelect: document.getElementById('user-select'),
            roleSelect: document.getElementById('role-select'),
            updateRoleBtn: document.getElementById('update-role-btn'),
            wsStatus: document.getElementById('ws-status'),
            wsQueue: document.getElementById('ws-queue'),
            volumeMeter: document.getElementById('volume-meter'),
            volumeText: document.getElementById('volume-text'),
            volumePeak: document.getElementById('volume-peak'),
            volumeAvg: document.getElementById('volume-avg'),
            exportMinutesBtn: document.getElementById('export-minutes-btn'),
            exportSpeechesBtn: document.getElementById('export-speeches-btn'),
            generateReportBtn: document.getElementById('generate-report-btn'),
            liveSummary: document.getElementById('live-summary'),
            speechTemplate: document.getElementById('speech-template'),
            participantTemplate: document.getElementById('participant-template'),
            statusMessage: document.getElementById('status-message')
        };
    }

    _setupEventListeners() {
        this.elements.recordBtn.addEventListener('click', this._toggleRecording.bind(this));
        this.elements.exportMinutesBtn.addEventListener('click', () => this._exportContent('minutes'));
        this.elements.exportSpeechesBtn.addEventListener('click', () => this._exportContent('speeches'));
        this.elements.generateReportBtn.addEventListener('click', () => this._exportContent('report'));
        this.elements.updateRoleBtn.addEventListener('click', this._updateUserRole.bind(this));
    }

    async _toggleRecording() {
        if (!this.isRecording) {
            this.elements.recordBtn.disabled = true;
            this.elements.recordBtn.classList.add('recording');
            this.elements.recordBtn.innerHTML = '<i class="bi bi-stop-fill"></i>';
            this.elements.permissionWarning.classList.add('d-none');

            try {
                await this._startMeetingProcesses();
                this.isRecording = true;
                this._updateMeetingStatus('进行中', 'success');
                this._resetMeetingUI();
                this._startFinalizationCheck();
            } catch (error) {
                this._handleRecordingError(error);
            }
        } else {
            this._stopRecording();
        }
    }

    _resetMeetingUI() {
        this.elements.recordBtn.classList.replace('btn-danger', 'btn-secondary');
        this.elements.recordBtn.disabled = false;
        this.elements.transcriptContainer.innerHTML = '';
        this.elements.participantsList.innerHTML = '';
        this.elements.userSelect.innerHTML = '<option value="">选择用户</option>';
        this.elements.currentSpeaker.textContent = '等待识别发言人...';
        this.elements.speakerConfidence.textContent = 'N/A';
        
        this.activeSpeaker = null;
        this.lastSpeakerId = null;
        this.lastAppendedEntry = null;
        this.lastTranscriptContent = "";
        this.lastReceivedTranscriptTime = 0;
        this.knownUsers.clear();
    }

    _handleRecordingError(error) {
        console.error("会议启动失败:", error);
        this.elements.recordBtn.classList.remove('recording');
        this.elements.recordBtn.innerHTML = '<i class="bi bi-mic"></i>';
        this.elements.recordBtn.classList.replace('btn-secondary', 'btn-danger');
        this._updateMeetingStatus('未开始', 'secondary');
        this.elements.recordBtn.disabled = false;

        const errorMessages = {
            'NotAllowedError': "麦克风权限被拒绝。请允许此网站访问您的麦克风，然后重试。",
            'NotFoundError': "未找到麦克风设备。请确保您的设备已连接麦克风。",
            'SecurityError': "安全错误。在非HTTPS环境下浏览器可能禁止访问麦克风。"
        };

        this.elements.permissionWarning.textContent = 
            errorMessages[error.name] || `启动失败: ${error.message || error.name || error}`;
        this.elements.permissionWarning.classList.remove('d-none');
        this._updateStatus("无法启动会议，请检查麦克风权限或控制台错误。", 'danger');
    }

    _stopRecording() {
        this.elements.recordBtn.disabled = true;
        this.elements.recordBtn.classList.remove('recording');
        this.elements.recordBtn.innerHTML = '<i class="bi bi-mic"></i>';
        this.elements.recordBtn.classList.replace('btn-secondary', 'btn-danger');

        this._stopMeetingProcesses();
        this.isRecording = false;
        this._updateMeetingStatus('已结束', 'info');
        this.elements.recordBtn.disabled = false;
        this._updateStatus("会议已结束。", 'info');
    }

    async _startMeetingProcesses() {
        this._updateStatus("会议启动中，尝试连接WebSocket...", 'info');
        this._updateConnectionStatus('连接中...', 0, 'warning');

        try {
            const meetingId = generateUUID();
            let clientId = localStorage.getItem('meetingClientId');
            if (!clientId) {
                clientId = generateUUID();
                localStorage.setItem('meetingClientId', clientId);
            }
            
            const WEBSOCKET_URL = `ws://localhost:8000/ws/meeting/${meetingId}/${clientId}`;
            console.log("尝试连接 WebSocket URL:", WEBSOCKET_URL);

            this.wsClient = new WebSocketClient(
                WEBSOCKET_URL,
                this._handleWebSocketMessage.bind(this),
                this._updateWebSocketStatus.bind(this)
            );
            await this.wsClient.connect();
            
            this.currentMeetingId = meetingId;
            this.currentClientId = clientId;

            this.audioRecorder = new AudioRecorder(
                this._handleVolumeData.bind(this),
                this.wsClient
            );
            await this.audioRecorder.start();
            
            this._updateStatus("麦克风和WebSocket启动成功！", 'success');
        } catch (error) {
            console.error("_startMeetingProcesses 内部错误:", error);
            if (this.wsClient) this.wsClient.disconnect();
            if (this.audioRecorder) this.audioRecorder.stop();
            this._updateStatus(`启动失败: ${error.message}`, 'danger');
            throw error;
        }
    }

    _stopMeetingProcesses() {
        console.log("会议停止逻辑：断开WebSocket，停止麦克风...");
        if (this.audioRecorder) {
            this.audioRecorder.stop();
            this.audioRecorder = null;
        }
        if (this.wsClient) {
            this.wsClient.disconnect();
            this.wsClient = null;
        }
        this._stopFinalizationCheck();

        if (this.lastAppendedEntry?.classList.contains('draft-transcript')) {
            this.lastAppendedEntry.classList.replace('draft-transcript', 'final-transcript');
        }

        this._updateConnectionStatus('断开', 0, 'danger');
        this._updateStatus("正在停止会议进程...", 'info');
    }

    _startFinalizationCheck() {
        this._stopFinalizationCheck();
        this.finalizationCheckInterval = setInterval(() => {
            const now = Date.now();
            if (this.activeSpeaker && this.lastAppendedEntry && 
                this.lastAppendedEntry.dataset.speakerId === this.activeSpeaker &&
                this.lastAppendedEntry.classList.contains('draft-transcript') &&
                (now - this.lastReceivedTranscriptTime > this.finalizationTimeout)) {
                
                console.log(`[Finalization Check] 将发言人 ${this.activeSpeaker} 的条目标记为最终。`);
                this.lastAppendedEntry.classList.replace('draft-transcript', 'final-transcript');
                this.lastAppendedEntry = null;
                this.activeSpeaker = null;
                this.lastSpeakerId = null;
            }
        }, this.finalizationTimeout / 2);
    }

    _stopFinalizationCheck() {
        if (this.finalizationCheckInterval) {
            clearInterval(this.finalizationCheckInterval);
            this.finalizationCheckInterval = null;
        }
    }

    _updateStatus(message, type = 'info') {
        if (this.elements.statusMessage) {
            this.elements.statusMessage.textContent = message;
            this.elements.statusMessage.className = `text-${type}`;
        }
    }

    _updateMeetingStatus(statusText, statusClass) {
        this.elements.meetingStatus.textContent = statusText;
        this.elements.meetingStatus.className = `badge bg-${statusClass}`;
    }

    _updateConnectionStatus(statusText, queueSize, statusClass) {
        this.elements.wsStatus.textContent = `连接状态: ${statusText}`;
        this.elements.wsQueue.textContent = `待发消息: ${queueSize}`;
        const container = this.elements.wsStatus.closest('.connection-status');
        if (container) container.className = `connection-status text-center mt-3 alert alert-${statusClass}`;
    }

    _handleVolumeData(currentVolume, peakVolume, averageVolume) {
        const updateElement = (id, value, formatter = v => v) => {
            const el = this.elements[id];
            if (el) el.textContent = formatter(value);
        };

        if (this.elements.volumeMeter) {
            this.elements.volumeMeter.style.width = `${currentVolume}%`;
            this.elements.volumeMeter.setAttribute('aria-valuenow', currentVolume);
            const volumeClass = currentVolume > 70 ? 'bg-danger' : 
                               currentVolume > 40 ? 'bg-warning' : 'bg-success';
            this.elements.volumeMeter.className = `progress-bar volume-indicator ${volumeClass}`;
        }

        updateElement('volumeText', currentVolume, v => v > 0 ? `${v}%` : '0% (未激活)');
        updateElement('volumePeak', peakVolume);
        updateElement('volumeAvg', averageVolume);
    }

    _updateSpeakerInfo(userName, confidence, role) {
        this.elements.currentSpeaker.textContent = userName || '等待识别发言人...';
        this.elements.speakerConfidence.textContent = `${(confidence * 100).toFixed(0)}%`;
        this.elements.speakerConfidence.className = `badge ${confidence > 0.7 ? 'bg-primary' : 'bg-secondary'}`;
    }

    _updateOrCreateTranscriptEntry(speakerName, transcript, timestamp) {
        const isNewSpeaker = this.lastSpeakerId && this.lastSpeakerId !== speakerName;
        const isLastEntryFinal = this.lastAppendedEntry?.classList.contains('final-transcript');

        if (isNewSpeaker || !this.lastAppendedEntry || 
            this.lastAppendedEntry.dataset.speakerId !== speakerName || isLastEntryFinal) {
            
            if (this.lastAppendedEntry?.classList.contains('draft-transcript')) {
                this.lastAppendedEntry.classList.replace('draft-transcript', 'final-transcript');
            }

            const template = this.elements.speechTemplate.content.cloneNode(true);
            const speechDiv = template.querySelector('.speech-bubble');
            speechDiv.querySelector('.speaker-name').textContent = speakerName;
            speechDiv.querySelector('.speaker-time').textContent = new Date(timestamp).toLocaleTimeString();
            speechDiv.querySelector('.speech-content').innerHTML = 
                typeof marked !== 'undefined' ? marked.parse(transcript) : transcript;
            
            speechDiv.classList.add('draft-transcript');
            speechDiv.dataset.speakerId = speakerName;
            
            this.elements.transcriptContainer.querySelector('.empty-state')?.remove();
            this.elements.transcriptContainer.appendChild(template);
            
            this.lastAppendedEntry = speechDiv;
            this.lastTranscriptContent = transcript;
        } else {
            const contentDiv = this.lastAppendedEntry.querySelector('.speech-content');
            if (contentDiv && this.lastTranscriptContent !== transcript) {
                contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(transcript) : transcript;
                this.lastTranscriptContent = transcript;
            }
        }
        
        this._scrollToBottom();
        this.lastSpeakerId = speakerName;
    }

    addOrUpdateParticipant(userName, role, lastActiveTime, speechCount) {
        const existing = this.elements.participantsList.querySelector(`[data-participant-name="${userName}"]`);
        
        if (existing) {
            existing.querySelector('.participant-role').textContent = role;
            existing.querySelector('.last-active').textContent = lastActiveTime;
            if (speechCount !== null) {
                existing.querySelector('.speech-count').textContent = `发言: ${speechCount}次`;
            }
        } else {
            const template = this.elements.participantTemplate.content.cloneNode(true);
            const item = template.querySelector('.list-group-item');
            item.dataset.participantName = userName;
            
            template.querySelector('.participant-name').textContent = userName;
            template.querySelector('.participant-role').textContent = role;
            template.querySelector('.last-active').textContent = lastActiveTime;
            template.querySelector('.speech-count').textContent = `发言: ${speechCount || 0}次`;
            
            this.elements.participantsList.appendChild(template);
            this.elements.participantCount.textContent = `${this.elements.participantsList.children.length}人`;

            const option = new Option(userName, userName);
            this.elements.userSelect.add(option);
        }
    }

    _updateLiveSummary(summaryText) {
        this.elements.liveSummary.innerHTML = 
            typeof marked !== 'undefined' ? marked.parse(summaryText) : summaryText;
    }

    _scrollToBottom() {
        this.elements.transcriptContainer.scrollTop = this.elements.transcriptContainer.scrollHeight;
    }

    _exportContent(type) {
        console.log(`执行 ${type} 导出`);
        alert(`功能待实现：导出 ${type}`);
    }

    _updateUserRole() {
        const userName = this.elements.userSelect.value;
        const newRole = this.elements.roleSelect.value;
        
        if (!userName || !newRole) {
            alert('请选择用户和角色！');
            return;
        }

        if (!this.wsClient?.isConnected()) {
            alert("WebSocket 未连接，无法更新角色。");
            return;
        }

        console.log(`尝试更新用户 ${userName} 的角色为 ${newRole}`);
        this.wsClient.send(JSON.stringify({
            type: "update_user_role",
            user_id: userName,
            new_role: newRole
        }));
    }

    _handleWebSocketMessage(message) {
        if (typeof message !== 'object' || !message.type) {
            console.warn("无效的WebSocket消息:", message);
            return;
        }

        console.log("收到消息:", message.type, message);

        switch (message.type) {
            case 'transcript_result':
                this._handleTranscriptMessage(message);
                break;
                
            case 'participant_list':
                this._handleParticipantList(message.participants);
                break;
                
            case 'speaker_identified':
                this._handleSpeakerIdentified(message);
                break;
                
            case 'recording_status':
                this._updateStatus(
                    message.status === 'started' ? 
                    "录音已开始，正在识别语音..." : "录音已停止。",
                    message.status === 'started' ? 'success' : 'info'
                );
                break;
                
            case 'error':
                console.error(`后端错误: ${message.message}`);
                this._updateStatus(`服务器错误: ${message.message}`, 'danger');
                break;
                
            case 'summary_update':
                this._updateLiveSummary(message.summary_text);
                break;
                
            case 'system_status_update':
                console.log("系统状态更新:", message.data);
                break;
                
            case 'meeting_init_response':
                console.log(`会议初始化成功！Meeting ID: ${message.meetingId}`);
                if (this.wsClient?.isConnected()) {
                    this.wsClient.send({ type: 'client_ready', clientId: message.clientId });
                }
                break;
        }
    }

    _handleTranscriptMessage(message) {
    // 确保有有效的用户名
    const userName = message.user_name || this.activeSpeaker || `${this.baseUserName}_${this.userCounter++}`;
    const transcript = message.transcript || "";
    const timestamp = message.timestamp || Date.now();
    const confidence = message.confidence || 0;
    const role = message.role || 'guest';

    // 更新当前说话人信息
    this._updateSpeakerInfo(userName, confidence, role);
    
    // 更新参与者列表
    if (!this.knownUsers.has(userName)) {
        this.knownUsers.set(userName, {userName, role});
        this.addOrUpdateParticipant(userName, role, new Date().toLocaleTimeString(), 0);
    }
    
    // 更新转录内容
    if (transcript.trim()) {
        this._updateOrCreateTranscriptEntry(userName, transcript, timestamp);
    }
    
    this.lastReceivedTranscriptTime = Date.now();
}

    _handleParticipantList(participants) {
        this.elements.participantsList.innerHTML = '';
        this.elements.userSelect.innerHTML = '<option value="">选择用户</option>';
        
        participants.forEach(participant => {
            this.addOrUpdateParticipant(
                participant.user_name,
                participant.role,
                new Date().toLocaleTimeString(),
                0
            );
        });
    }

    _handleSpeakerIdentified(message) {
    // 确保至少有默认用户名
    const userName = message.user_name || `${this.baseUserName}_${this.userCounter++}`;
    const confidence = message.confidence || 0;
    const role = message.role || 'guest';
    
    this.activeSpeaker = userName;
    this._updateSpeakerInfo(userName, confidence, role);
    
    // 添加到参与者列表
    if (!this.knownUsers.has(userName)) {
        this.knownUsers.set(userName, {userName, role});
        this.addOrUpdateParticipant(userName, role, new Date().toLocaleTimeString(), 0);
    }
   }

    _updateWebSocketStatus(statusText, queueSize, statusClass) {
        this._updateConnectionStatus(statusText, queueSize, statusClass);
    }
}
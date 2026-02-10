// M:\meeting\client-app\src\App.jsx (或 App.js)

// 引入 React 核心库及其 Hooks
import React, { useState, useEffect, useRef, useCallback, createContext, useContext } from 'react';
import VoiceLogin from './VoiceLogin'; // 导入声纹登录组件
// 引入 Lucide React 图标库
import { Mic, User, Users, Edit, Book, Download, MessageSquare, RefreshCw, XCircle, CheckCircle, Info, Send, Volume2, VolumeX, Loader, Settings, PlusCircle, MinusCircle } from 'lucide-react';

// --- 全局状态管理上下文 ---
// 创建一个 React Context，用于在组件树中共享状态，避免逐层传递 props
const AppContext = createContext(null);

// --- 工具函数 ---
// 简单的防抖函数，用于限制 API 调用频率
const debounce = (func, delay) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout); // 清除之前的定时器
        timeout = setTimeout(() => func(...args), delay); // 设置新的定时器
    };
};

// --- Backend API Configuration ---
// In production, use the current origin. In dev, use localhost:8000 or VITE env.
const API_BASE_URL = import.meta.env.PROD
    ? window.location.origin
    : (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000');


// --- Main App Component ---
// This is the root component of the React application
function App() {
    // --- State Definitions ---
    const [meetingId, setMeetingId] = useState(() => {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('meetingId') || `meeting-${Date.now()}`;
    });
    const [clientId, setClientId] = useState(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const voiceLoginId = urlParams.get('voiceLoginId'); // Get voice login ID from URL
        if (voiceLoginId) {
            localStorage.setItem('clientId', voiceLoginId); // Store voice login ID
            return voiceLoginId;
        }
        const storedClientId = localStorage.getItem('clientId');
        return storedClientId || `client-${Math.random().toString(36).substring(2, 9)}`;
    });
    const [isLoggedIn, setIsLoggedIn] = useState(localStorage.getItem('isVoiceAuthenticated') === 'true'); // Check login status based on voice auth

    // Callback for successful voice login
    const handleLoginSuccess = useCallback((userData) => {
        setIsLoggedIn(true);
        setClientId(userData._id); // Use the User ID returned from voice login
        localStorage.setItem('clientId', userData._id); // Store new clientId
        localStorage.setItem('isVoiceAuthenticated', 'true'); // Set voice auth flag
        // Redirect or refresh to clear voiceLoginId from URL and ensure all components use the new clientId
        window.history.replaceState({}, document.title, window.location.pathname);
    }, []);

    // Logout function
    const handleLogout = useCallback(() => {
        localStorage.removeItem('clientId');
        localStorage.removeItem('isVoiceAuthenticated');
        setIsLoggedIn(false);
        // Refresh page to reset state and show login screen
        window.location.reload();
    }, []);
    const [isConnected, setIsConnected] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [transcriptData, setTranscriptData] = useState({});
    const [users, setUsers] = useState([]);
    const [activeSpeakers, setActiveSpeakers] = useState(new Set()); // Use Set for active speakers
    const [systemStatus, setSystemStatus] = useState({});
    const [showEditModal, setShowEditModal] = useState(false);
    const [editingUser, setEditingUser] = useState(null);
    const [llmQAHistory, setLlmQAHistory] = useState([]);
    const [llmQALoading, setLlmQALoading] = useState(false);
    const [audioMonitorLevel, setAudioMonitorLevel] = useState(0);
    const [audioContext, setAudioContext] = useState(null);
    const [mediaStreamSource, setMediaStreamSource] = useState(null);
    const [analyserNode, setAnalyserNode] = useState(null);
    const [audioProcessor, setAudioProcessor] = useState(null);
    const [isMicMuted, setIsMicMuted] = useState(false);

    // --- Ref Definitions ---
    const wsRef = useRef(null);
    const mediaStreamRef = useRef(null);

    // --- API Calls (defined before functions dependent on them) ---
    // Get all users
    const fetchUsers = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/get-all-users`);
            if (!response.ok) {
                throw new Error(`HTTP Error! Status: ${response.status}`);
            }
            const data = await response.json();
            setUsers(data); // Update user list state
            console.log('Users fetched:', data);
        } catch (error) {
            console.error('Failed to fetch users:', error);
            // Optionally show error message to user
        }
    }, []); // Empty dependency array means create once on mount

    // Get system status
    const fetchSystemStatus = useCallback(async () => {
        try {
            const statusResponse = await fetch(`${API_BASE_URL}/system/status`);
            if (!statusResponse.ok) {
                // throw new Error(`HTTP Error! Status: ${statusResponse.status}`);
                // Silently fail or log
                return;
            }
            const data = await statusResponse.json();
            setSystemStatus(data); // Update system status
        } catch (error) {
            console.error('Failed to fetch system status:', error);
        }
    }, []);


    // --- WebSocket Connection Management ---
    const connectWebSocket = useCallback(() => {
        // Do not reconnect if WebSocket is already open
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            console.log('WebSocket is already connected.');
            return;
        }

        // WebSocket Connection URL
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Dynamic WS Host
        const wsHost = import.meta.env.PROD
            ? window.location.host
            : (import.meta.env.VITE_WS_HOST || '127.0.0.1:8000');

        const wsUrl = `${wsProtocol}//${wsHost}/ws/meeting/${meetingId}/${clientId}`;

        console.log(`Attempting to connect WebSocket to ${wsUrl}`);
        // 创建新的 WebSocket 实例
        const newWs = new WebSocket(wsUrl);
        newWs.binaryType = 'arraybuffer'; // 显式设置二进制类型，确保接收二进制数据为 ArrayBuffer

        // WebSocket 连接成功事件
        newWs.onopen = () => {
            console.log('WebSocket 连接成功。');
            setIsConnected(true); // 更新连接状态
            wsRef.current = newWs; // 直接在 Ref 中设置 WebSocket 实例
            localStorage.setItem('clientId', clientId); // 将客户端 ID 存储到本地存储
            // 向后端发送 'client_ready' 消息
            newWs.send(JSON.stringify({ type: 'client_ready' }));
        };

        // WebSocket 接收消息事件
        newWs.onmessage = async (event) => {
            let messageData = event.data;

            // 如果接收到的是 Blob，尝试将其读取为文本
            if (messageData instanceof Blob) {
                try {
                    messageData = await messageData.text(); // 将 Blob 读取为文本
                } catch (e) {
                    console.error("Failed to read Blob as text:", e);
                    return; // 如果 Blob 无法读取，则无法处理此消息
                }
            } else if (messageData instanceof ArrayBuffer) {
                // 如果收到 ArrayBuffer，尝试将其解码为 UTF-8 字符串
                try {
                    messageData = new TextDecoder("utf-8").decode(messageData);
                } catch (e) {
                    console.error("Failed to decode ArrayBuffer as text:", e);
                    return;
                }
            }

            // 现在，messageData 应该是一个字符串
            if (typeof messageData === 'string') {
                // 忽略后端发送的 'ping' 心跳消息
                if (messageData === 'ping') {
                    return;
                }
                try {
                    const message = JSON.parse(messageData);
                    switch (message.type) {
                        case 'meeting_init_response':
                            console.log('会议已初始化:', message);
                            if (message.userId && message.userId !== clientId) {
                                setClientId(message.userId);
                                localStorage.setItem('clientId', message.userId);
                            }
                            // 在收到初始化响应后，拉取所有用户列表
                            fetchUsers();
                            break;
                        case 'transcript_update':
                            setTranscriptData(prev => {
                                const newSpeakerData = prev[message.speakerId] ? [...prev[message.speakerId]] : [];
                                const lastEntry = newSpeakerData[newSpeakerData.length - 1];
                                if (lastEntry && !lastEntry.isFinal && message.isFinal === false) {
                                    newSpeakerData[newSpeakerData.length - 1] = {
                                        text: message.text,
                                        timestamp: message.timestamp,
                                        isFinal: message.isFinal
                                    };
                                } else {
                                    newSpeakerData.push({
                                        text: message.text,
                                        timestamp: message.timestamp,
                                        isFinal: message.isFinal
                                    });
                                }
                                return { ...prev, [message.speakerId]: newSpeakerData };
                            });
                            break;
                        case 'user_joined':
                            console.log('WS: 新用户加入:', message.user);
                            setUsers(prevUsers => {
                                // 检查用户是否已存在以避免重复添加
                                if (!prevUsers.some(u => u._id === message.user._id)) {
                                    // 如果不存在，将新用户添加到列表中
                                    return [...prevUsers, message.user];
                                }
                                return prevUsers;
                            });
                            break;
                        case 'speaker_identified':
                            console.log('发言人已识别:', message);
                            setUsers(prevUsers => {
                                const existingUser = prevUsers.find(u => u._id === message.userId);
                                if (!existingUser) {
                                    return [...prevUsers, { _id: message.userId, name: message.name, role: message.role }];
                                }
                                return prevUsers;
                            });
                            break;
                        case 'user_role_updated':
                            console.log('用户角色已更新:', message);
                            setUsers(prevUsers => prevUsers.map(u =>
                                u._id === message._id ? { ...u, role: message.new_role } : u
                            ));
                            break;
                        case 'system_status_update':
                            setSystemStatus(message.data);
                            break;
                        case 'error':
                            console.error('WS 错误:', message.message);
                            break;
                        default:
                            console.warn('未知 WS 消息类型:', message.type, message);
                    }
                } catch (e) {
                    console.error('WebSocket 消息解析失败 (非 JSON 字符串):', e, '原始数据:', messageData);
                }
            } else {
                console.warn('收到非字符串、非 Blob、非 ArrayBuffer 类型的 WebSocket 消息。忽略。', messageData);
            }
        };

        // WebSocket 连接关闭事件
        newWs.onclose = (event) => {
            console.warn('WebSocket 已断开连接:', event.code, event.reason);
            setIsConnected(false); // 更新连接状态
            wsRef.current = null; // 清除 Ref 中的 WebSocket 实例
            // 延迟后尝试重新连接
            setTimeout(connectWebSocket, 3000);
        };

        // WebSocket 错误事件
        newWs.onerror = (error) => {
            console.error('WebSocket 错误:', error);
            newWs.close(); // 关闭连接以触发 onclose 并尝试重新连接
        };
    }, [meetingId, clientId, fetchUsers]); // 依赖项：meetingId, clientId, fetchUsers

    // --- 麦克风音频输入与处理 ---
    const startMicrophone = useCallback(async () => {
        console.log("Attempting to start microphone...");
        // 如果正在录音或已静音，则不执行
        if (isRecording || isMicMuted) {
            console.log(`startMicrophone aborted. isRecording: ${isRecording}, isMicMuted: ${isMicMuted}`);
            return;
        }

        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('浏览器不支持访问麦克风。请确保在安全上下文（HTTPS）中运行。');
            }
            // 获取用户麦克风媒体流
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log("Microphone access granted (getUserMedia success).");
            mediaStreamRef.current = stream; // 保存媒体流引用

            // Create Web Audio API context with 16kHz sample rate to match backend requirements
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            setAudioContext(audioCtx);
            const originalSampleRate = audioCtx.sampleRate; // Should be 16000 if supported, otherwise hardware rate
            console.log(`AudioContext initialized with sample rate: ${originalSampleRate}Hz`);

            // Create media stream source node
            const source = audioCtx.createMediaStreamSource(stream);
            setMediaStreamSource(source);

            // Create analyzer node
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            setAnalyserNode(analyser);

            // --- Code to dynamically create AudioWorklet module ---
            const audioProcessorCode = `
            const resampleAudio = (audioBuffer, originalSampleRate, targetSampleRate) => {
                if (originalSampleRate === targetSampleRate) {
                    return audioBuffer;
                }
                const ratio = targetSampleRate / originalSampleRate;
                const newLength = Math.round(audioBuffer.length * ratio);
                const result = new Float32Array(newLength);
                const offset = Math.min(audioBuffer.length - 1, Math.floor(1 / ratio));
                for (let i = 0; i < newLength; i++) {
                    const index = Math.min(audioBuffer.length - 1, Math.floor(i / ratio));
                    const a = audioBuffer[index];
                    const b = audioBuffer[Math.min(audioBuffer.length - 1, index + offset)];
                    const fraction = (i / ratio) - index;
                    result[i] = a + fraction * (b - a);
                }
                return result;
            };

            class AudioProcessor extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.originalSampleRate = 0;
                    this.targetSampleRate = 16000;
                    this.bufferSize = 4096; // Fixed buffer size (approx 256ms at 16k)
                    this.buffer = new Float32Array(this.bufferSize);
                    this.bufferIndex = 0;

                    this.port.onmessage = (event) => {
                        if (event.data.type === 'init') {
                            this.originalSampleRate = event.data.originalSampleRate;
                            this.targetSampleRate = event.data.targetSampleRate;
                        }
                    };
                }

                process(inputs, outputs, parameters) {
                    const input = inputs[0];
                    if (!input || input.length === 0) {
                        return true;
                    }
                    const inputBuffer = input[0];
                    let processedData = inputBuffer;

                    if (this.originalSampleRate !== this.targetSampleRate) {
                        processedData = resampleAudio(inputBuffer, this.originalSampleRate, this.targetSampleRate);
                    }
                    
                    // Buffer logic: accumulate samples until we reach bufferSize
                    for (let i = 0; i < processedData.length; i++) {
                        this.buffer[this.bufferIndex++] = processedData[i];
                        
                        if (this.bufferIndex >= this.bufferSize) {
                            // Flush buffer
                            const pcm16 = new Int16Array(this.bufferSize);
                            for (let j = 0; j < this.bufferSize; j++) {
                                pcm16[j] = Math.max(-1, Math.min(1, this.buffer[j])) * 0x7FFF;
                            }
                            this.port.postMessage({ type: 'audioData', pcm16Data: pcm16 });
                            this.bufferIndex = 0; // Reset buffer index
                        }
                    }
                    return true;
                }
            }

            registerProcessor('audio-processor', AudioProcessor);
        `;
            const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
            const blobUrl = URL.createObjectURL(blob);
            await audioCtx.audioWorklet.addModule(blobUrl);
            console.log("AudioWorklet module added successfully.");
            // --- 动态创建 AudioWorklet 模块的代码结束 ---

            const audioWorkletNode = new AudioWorkletNode(audioCtx, 'audio-processor');

            const VOICE_SAMPLE_RATE = 16000;
            audioWorkletNode.port.postMessage({
                type: 'init',
                originalSampleRate,
                targetSampleRate: VOICE_SAMPLE_RATE
            });

            // AudioWorkletNode 通过 port 发送处理后的数据
            audioWorkletNode.port.onmessage = (event) => {
                console.log("Received message from AudioWorklet.");
                const { type, pcm16Data } = event.data;
                if (type === 'audioData' && wsRef.current && wsRef.current.readyState === WebSocket.OPEN && !isMicMuted) {
                    console.log(`Sending audio data as Blob... (size: ${pcm16Data.buffer.byteLength})`);
                    // 通过 WebSocket 发送音频数据
                    wsRef.current.send(new Blob([pcm16Data.buffer]));
                } else {
                    console.warn("Condition to send audio data not met:", {
                        isAudioData: type === 'audioData',
                        isWsOpen: wsRef.current && wsRef.current.readyState === WebSocket.OPEN,
                        isNotMuted: !isMicMuted
                    });
                }
            };

            // 连接音频节点：源 -> 分析器 -> AudioWorkletNode -> 音频输出
            source.connect(analyser);
            analyser.connect(audioWorkletNode);
            audioWorkletNode.connect(audioCtx.destination); // 连接到音频输出，保持处理链活跃

            // --- 新增的音量监控逻辑 ---
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            const getAudioLevel = () => {
                analyser.getByteFrequencyData(dataArray);
                // 计算音频级别（简单地取平均值）
                const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
                const normalizedLevel = (average / 256) * 100; // 归一化为 0-100 的百分比

                // 更新状态
                setAudioMonitorLevel(normalizedLevel);

                // 循环调用
                requestAnimationFrame(getAudioLevel);
            };
            // 在麦克风启动后立即开始监控
            requestAnimationFrame(getAudioLevel);
            // --- 新增的音量监控逻辑结束 ---

            setIsRecording(true); // 更新录音状态
            // 向后端发送 'start_recording' 消息
            if (wsRef.current) {
                wsRef.current.send(JSON.stringify({ type: 'start_recording' }));
            }
            console.log('麦克风已启动。');
        } catch (error) {
            console.error('访问麦克风出错:', error);
            // 删除了阻塞的 alert()，用 console.error 替换。
            setIsRecording(false); // 启动失败则重置状态
        }
    }, [isRecording, isMicMuted]);

    const stopMicrophone = useCallback(() => {
        if (!isRecording) return; // 如果未录音，则不执行

        // 停止媒体流的所有轨道
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(track => track.stop());
            mediaStreamRef.current = null;
        }
        // 断开并清理音频处理器节点
        if (audioProcessor) {
            audioProcessor.disconnect();
            setAudioProcessor(null);
        }
        // 断开并清理分析器节点
        if (analyserNode) {
            analyserNode.disconnect();
            setAnalyserNode(null);
        }
        // 断开并清理媒体流源节点
        if (mediaStreamSource) {
            mediaStreamSource.disconnect();
            setMediaStreamSource(null);
        }
        // 关闭音频上下文
        if (audioContext) {
            audioContext.close();
            setAudioContext(null);
        }

        setIsRecording(false); // 更新录音状态
        setAudioMonitorLevel(0); // 重置音频监控级别
        // 向后端发送 'stop_recording' 消息
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'stop_recording' }));
        }
        console.log('麦克风已停止。');
    }, [isRecording, audioProcessor, analyserNode, mediaStreamSource, audioContext]); // 依赖项

    const toggleMicrophone = useCallback(() => {
        if (isRecording) {
            stopMicrophone();
        } else {
            startMicrophone();
        }
    }, [isRecording, startMicrophone, stopMicrophone]);

    const toggleMute = useCallback(() => {
        if (mediaStreamRef.current) {
            // 遍历所有音频轨道并切换其启用状态
            mediaStreamRef.current.getAudioTracks().forEach(track => {
                track.enabled = isMicMuted; // 如果当前是静音，则启用；如果已启用，则静音
            });
        }
        setIsMicMuted(prev => {
            const newState = !prev; // 切换静音状态
            if (newState && isRecording) { // 如果正在静音并且正在录音，则停止发送音频
                console.log("麦克风已静音。停止音频传输。");
                // 这里不需要完全停止麦克风，只需在 onaudioprocess 中停止发送数据
            } else if (!newState && !isRecording) { // 如果取消静音并且未录音，则开始录音
                console.log("麦克风已取消静音。开始音频传输。");
                startMicrophone(); // 重新启动麦克风以发送数据
            }
            return newState;
        });
    }, [isMicMuted, isRecording, startMicrophone]); // 依赖项

    // --- 初始化设置和清理 (依赖所有上述函数，所以定义在它们之后) ---
    useEffect(() => {
        connectWebSocket(); // 连接 WebSocket
        fetchUsers(); // 获取所有用户
        fetchSystemStatus(); // 获取系统状态

        // 每隔 5 秒轮询一次系统状态
        const systemStatusInterval = setInterval(fetchSystemStatus, 5000);

        // 组件卸载时的清理函数
        return () => {
            if (wsRef.current) {
                wsRef.current.close(); // 关闭 WebSocket 连接
            }
            stopMicrophone(); // 停止麦克风
            clearInterval(systemStatusInterval); // 清除系统状态轮询定时器
        };
    }, [connectWebSocket, stopMicrophone, fetchUsers, fetchSystemStatus]); // 依赖项：确保这些函数是最新的

    // --- 处理活跃发言人状态
    // 这个 useEffect 专门用于在 transcriptData 变化时更新 activeSpeakers
    useEffect(() => {
        const activeSpeakersTimeout = {}; // 用于存储每个发言人的定时器

        // 清理旧的定时器
        Object.keys(activeSpeakersTimeout).forEach(speakerId => {
            clearTimeout(activeSpeakersTimeout[speakerId]);
        });

        // 遍历 transcriptData，为每个发言人设置活跃状态和定时器
        Object.entries(transcriptData).forEach(([speakerId, transcripts]) => {
            if (transcripts.length > 0) {
                const lastEntry = transcripts[transcripts.length - 1];
                // 如果是最终转录，或者非最终转录但最近有更新，则标记为活跃
                if (lastEntry.isFinal || (Date.now() - new Date(lastEntry.timestamp).getTime() < 2000)) { // 2秒内有更新
                    setActiveSpeakers(prev => {
                        const newSet = new Set(prev);
                        newSet.add(speakerId);
                        return newSet;
                    });
                    // 设置一个定时器，在一段时间后移除活跃状态
                    activeSpeakersTimeout[speakerId] = setTimeout(() => {
                        setActiveSpeakers(prev => {
                            const newSet = new Set(prev);
                            newSet.delete(speakerId);
                            return newSet;
                        });
                    }, 3000); // 3秒后移除活跃状态
                }
            }
        });

        // 组件卸载时清理所有定时器
        return () => {
            Object.keys(activeSpeakersTimeout).forEach(speakerId => {
                clearTimeout(activeSpeakersTimeout[speakerId]);
            });
        };
    }, [transcriptData]); // 依赖 transcriptData 变化

    // --- 其他 API 调用处理函数 ---
    const handleUpdateUserRole = async (userId, newUsername, newRole) => {
        try {
            const response = await fetch(`${API_BASE_URL}/update-role?voiceprint_id=${userId}&new_role=${newRole}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP 错误! 状态码: ${response.status}`);
            }
            const data = await response.json();
            console.log('用户角色已更新:', data);
            // 更新本地用户列表状态
            setUsers(prevUsers => prevUsers.map(u =>
                u._id === userId ? { ...u, name: newUsername, role: newRole } : u
            ));
            setShowEditModal(false); // 关闭编辑模态框
            setEditingUser(null); // 清除正在编辑的用户
            // 通过 WebSocket 通知后端，或依赖后端广播更新
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                    type: 'update_user_role',
                    voiceprint_id: userId,  // 改为voiceprint_id
                    new_role: newRole,
                    new_name: newName       // 保持new_name
                }));
            }
        } catch (error) {
            console.error('更新用户角色失败:', error);
            alert(`更新用户角色失败: ${error.message}`);
        }
    };

    const handleGenerateMinutes = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/generate_minutes?meeting_id=${meetingId}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP 错误! 状态码: ${response.status}`);
            }
            // 后端返回的是 docx 文件流，需要处理为 Blob 并触发下载
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `会议纪要-${meetingId}.docx`; // 设置下载文件名
            document.body.appendChild(a);
            a.click(); // 触发点击下载
            a.remove(); // 移除临时创建的链接
            window.URL.revokeObjectURL(url); // 释放 Blob URL
            alert('会议纪要已生成并开始下载！');
        } catch (error) {
            console.error('生成会议纪要失败:', error);
            alert(`生成会议纪要失败: ${error.message}`);
        }
    };

    const handleGenerateReport = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/generate_meeting_report?meeting_id=${meetingId}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP 错误! 状态码: ${response.status}`);
            }
            const data = await response.json();
            alert(`会议总结报告生成成功！内容已在控制台打印，或您可以在LLM问答窗口查看。\n\n${data.report.content.substring(0, 200)}...`);
            console.log('生成的会议报告:', data.report.content);
            // 将报告内容添加到 LLM 问答历史中显示
            setLlmQAHistory(prev => [...prev, { type: 'ai', text: '会议总结报告已生成：\n' + data.report.content }]);
        } catch (error) {
            console.error('生成报告失败:', error);
            alert(`生成会议总结报告失败: ${error.message}`);
        }
    };

    const handleExportUserSpeech = async (targetUserId = null) => {
        try {
            const url = targetUserId
                ? `${API_BASE_URL}/export_user_speech?meeting_id=${meetingId}&_id=${targetUserId}`
                : `${API_BASE_URL}/export_user_speech?meeting_id=${meetingId}`;

            const response = await fetch(url);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP 错误! 状态码: ${response.status}`);
            }
            const blob = await response.blob();
            const filename = targetUserId ? `用户发言-${targetUserId}-${meetingId}.txt` : `所有发言-${meetingId}.txt`;
            const urlBlob = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = urlBlob;
            a.download = filename; // 设置下载文件名
            document.body.appendChild(a);
            a.click(); // 触发点击下载
            a.remove(); // 移除临时创建的链接
            window.URL.revokeObjectURL(urlBlob); // 释放 Blob URL
            alert('用户发言内容已导出并开始下载！');
        } catch (error) {
            console.error('导出用户发言失败:', error);
            alert(`导出用户发言失败: ${error.message}`);
        }
    };

    const handleLinkMeetingToKG = async () => {
        try {
            alert('开始抽取知识图谱三元组，这可能需要一些时间...');
            const response = await fetch(`${API_BASE_URL}/link_meeting_to_kg?meeting_id=${meetingId}`, {
                method: 'POST',
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP 错误! 状态码: ${response.status}`);
            }
            const data = await response.json();
            console.log('知识图谱抽取结果:', data);
            alert('知识图谱三元组抽取完成！');
            setLlmQAHistory(prev => [...prev, { type: 'ai', text: '知识图谱三元组抽取完成！\n' + JSON.stringify(data, null, 2) }]);

        } catch (error) {
            console.error('链接会议到知识图谱失败:', error);
            alert(`知识图谱抽取失败: ${error.message}`);
        }
    };

    const handleLLMQuestion = async (question) => {
        // 1. 设置加载状态并立即添加用户问题到历史记录
        setLlmQALoading(true);
        setLlmQAHistory(prev => [...prev, { type: 'user', text: question }]);

        // 2. 在用户问题后立即为 AI 响应添加一个空的占位符条目，以便流式更新

        let aiResponsePlaceholderIndex = -1;
        setLlmQAHistory(prev => {
            const newHistory = [...prev, { type: 'ai', text: '' }];
            aiResponsePlaceholderIndex = newHistory.length - 1;
            return newHistory;
        });

        try {
            // 3. 发送 POST 请求到后端 LLM 流式问答 API
            const response = await fetch(`${API_BASE_URL}/ask_llm_stream/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                }),
            });

            // 4. 检查响应状态和流是否存在
            if (!response.ok || !response.body) {
                const errorText = await response.text();
                throw new Error(`HTTP 错误! 状态码: ${response.status} - ${errorText}`);
            }

            // 5. 核心：使用 TextDecoder 和 reader 处理流式响应
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let done = false;

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                done = readerDone;

                if (value) {
                    const chunk = decoder.decode(value, { stream: true });
                    // 6. 实时更新 AI 响应的占位符条目
                    setLlmQAHistory(prev => {
                        const newHistory = [...prev];
                        const currentText = newHistory[aiResponsePlaceholderIndex].text;
                        newHistory[aiResponsePlaceholderIndex].text = currentText + chunk;
                        return newHistory;
                    });
                }
            }

        } catch (error) {
            console.error('LLM 流式问答失败:', error);
            // 7. 错误处理：如果 AI 占位符为空，则更新它；否则添加新的错误消息
            setLlmQAHistory(prev => {
                const newHistory = [...prev];
                const errorText = `抱歉，回答问题时发生错误: ${error.message}`;
                if (aiResponsePlaceholderIndex !== -1 && newHistory[aiResponsePlaceholderIndex].text === '') {
                    newHistory[aiResponsePlaceholderIndex].text = errorText;
                } else {
                    newHistory.push({ type: 'ai', text: errorText });
                }
                return newHistory;
            });
        } finally {
            // 8. 无论成功与否，最后都结束加载状态
            setLlmQALoading(false);
        }
    };

    // --- Context 值 ---
    const appContextValue = {
        meetingId,
        clientId,
        isConnected,
        isRecording,
        toggleMicrophone,
        isMicMuted,
        toggleMute,
        transcriptData,
        users,
        activeSpeakers,
        systemStatus,
        setShowEditModal,
        setEditingUser,
        handleGenerateMinutes,
        handleGenerateReport,
        handleExportUserSpeech,
        handleLinkMeetingToKG,
        llmQAHistory,
        llmQALoading,
        handleLLMQuestion,
        audioMonitorLevel,
        fetchUsers,
    };

    // --- 主应用组件的渲染部分 ---
    return (
        <AppContext.Provider value={appContextValue}>
            {isLoggedIn ? (
                // 用户已登录，显示会议助手主界面
                <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100 font-inter antialiased p-4">
                    {/* 顶部头部区域 */}
                    <header className="bg-gray-950 p-4 rounded-xl shadow-lg border-b border-gray-700 mb-6">
                        <div className="container mx-auto flex justify-between items-center">
                            <h1 className="text-3xl font-bold text-blue-400 drop-shadow-md">
                                AI会议助手
                            </h1>
                            <div className="flex items-center space-x-4 text-sm">
                                {/* 会议 ID 显示 */}
                                <span className="flex items-center">
                                    <Info className="w-4 h-4 mr-1 text-blue-300" />
                                    会议ID: <span className="font-mono text-blue-200 ml-1">{meetingId}</span>
                                </span>
                                {/* 您的客户端 ID 显示 */}
                                <span className="flex items-center">
                                    <User className="w-4 h-4 mr-1 text-blue-300" />
                                    您的ID: <span className="font-mono text-blue-200 ml-1">{clientId}</span>
                                </span>
                                {/* 连接状态指示器 */}
                                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${isConnected ? 'bg-green-600' : 'bg-red-600'} transition-colors duration-300`}>
                                    {isConnected ? '在线' : '离线'}
                                </span>
                                {/* 登出按钮 */}
                                <button
                                    onClick={handleLogout}
                                    className="px-3 py-1 rounded-full text-xs font-semibold bg-red-600 hover:bg-red-700 transition-colors duration-300"
                                >
                                    登出
                                </button>
                            </div>
                        </div>
                    </header>

                    {/* 主内容网格布局 */}
                    <main className="container mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* 左侧/主列：音频监控和发言人转录窗口 */}
                        <div className="lg:col-span-2 space-y-6">
                            <MicrophoneMonitor /> {/* 麦克风监控组件 */}
                            <SpeakerTranscriptWindow /> {/* 发言人转录窗口组件 */}
                        </div>

                        {/* 右侧边栏：用户列表、控制面板、LLM 问答、系统状态 */}
                        <aside className="space-y-6">
                            <UserList /> {/* 用户列表组件 */}
                            <ControlPanel /> {/* 控制面板组件 */}
                            <LLMQAWindow /> {/* LLM 问答窗口组件 */}
                            <SystemStatusDisplay /> {/* 系统状态显示组件 */}
                        </aside>
                    </main>

                    {/* 编辑用户模态框 (条件渲染) */}
                    {showEditModal && editingUser && (
                        <SpeakerEditModal
                            user={editingUser}
                            onClose={() => setShowEditModal(false)}
                            onSave={handleUpdateUserRole}
                        />
                    )}
                </div>
            ) : (
                // 用户未登录，显示声纹登录组件
                <VoiceLogin onLoginSuccess={handleLoginSuccess} />
            )}
        </AppContext.Provider>
    );
}

// --- 麦克风监控组件 ---
function MicrophoneMonitor() {
    const { isRecording, toggleMicrophone, isMicMuted, toggleMute, audioMonitorLevel } = useContext(AppContext);
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        //  根据音频级别计算条形的宽度
        const barWidth = audioMonitorLevel * width / 100;

        // *** 渐变方向从左到右
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#0f766e'); // 青色-700 (现在在左侧低音量)
        gradient.addColorStop(0.5, '#2dd4bf'); // 青色-400
        gradient.addColorStop(1, '#a78bfa'); // 紫罗兰色-400 (用于更高音量，现在在右侧高音量)

        ctx.fillStyle = gradient;
        // ** 从左上角 (X=0, Y=0) 开始填充，宽度为 barWidth，高度为整个 canvas 高度
        ctx.fillRect(0, 0, barWidth, height);

        // 添加一个细微的边框
        ctx.strokeStyle = '#374151'; // 灰色-700
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, width, height);

    }, [audioMonitorLevel]);

    return (
        <div className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 flex flex-col items-center">

            <div className="w-full flex items-center justify-center space-x-4">
                {/* 录音/停止按钮 */}
                <button
                    onClick={toggleMicrophone}
                    className={`p-3 rounded-full shadow-md transition-all duration-300 ease-in-out
                                ${isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
                                focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 ${isRecording ? 'focus:ring-red-500' : 'focus:ring-green-500'}`}
                    title={isRecording ? '停止录音' : '开始录音'}
                >
                    <Mic className="w-6 h-6 text-white" />
                </button>
                {/* 静音/取消静音按钮 */}
                <button
                    onClick={toggleMute}
                    className={`p-3 rounded-full shadow-md transition-all duration-300 ease-in-out
                                ${isMicMuted ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-blue-600 hover:bg-blue-700'}
                                focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 ${isMicMuted ? 'focus:ring-yellow-500' : 'focus:ring-blue-500'}`}
                    title={isMicMuted ? '取消静音' : '静音麦克风'}
                >
                    {isMicMuted ? <VolumeX className="w-6 h-6 text-white" /> : <Volume2 className="w-6 h-6 text-white" />}
                </button>
            </div>
            {/* 音频可视化区域 */}
            {/* ** 减小高度，改为横向增长 */}
            <div className="w-full h-8 bg-gray-900 rounded-lg mt-4 overflow-hidden relative border border-gray-700">
                {/* Canvas 用于绘制更复杂的音频波形，这里用作背景 */}
                <canvas ref={canvasRef} width="300" height="32" className="absolute top-0 left-0 w-full h-full"></canvas>
                {/* 使用 div 模拟的音频强度条，与 canvas 效果类似，但更简单 */}
                <div
                    // ** 使用 top-0 和 left-0 定位，宽度动态变化，高度 100% 充满父容器
                    className="absolute top-0 left-0 bg-gradient-to-r from-teal-700 via-teal-400 to-violet-400 transition-all duration-100 ease-out"
                    style={{ width: `${audioMonitorLevel}%`, height: '100%' }} // 高度 100% 充满，宽度动态变化
                ></div>
            </div>
            <p className="text-sm text-gray-400 mt-2">音频强度: {audioMonitorLevel.toFixed(0)}%</p>
        </div>
    );
}

// --- 发言人转录窗口组件 ---
function SpeakerTranscriptWindow() {
    // 从上下文中获取转录数据和活跃发言人
    const { transcriptData, activeSpeakers } = useContext(AppContext);
    // useRef 用于存储每个发言人转录容器的引用，以便滚动
    const scrollRefs = useRef({});

    // useEffect 在转录数据变化时滚动到最新发言
    useEffect(() => {
        Object.keys(transcriptData).forEach(speakerId => {
            const ref = scrollRefs.current[speakerId];
            if (ref) {
                ref.scrollTop = ref.scrollHeight; // 滚动到底部
            }
        });
    }, [transcriptData]); // 依赖项：当 transcriptData 变化时执行

    return (
        <div className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700">
            {/* **修改点 2：将“发言人窗口”标题改为图标显示** */}
            <h2 className="text-xl font-semibold text-blue-300 mb-3 flex items-center">
                <MessageSquare className="w-6 h-6 mr-2 text-blue-400" />

            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto custom-scrollbar">
                {/* 如果没有转录数据，显示提示信息 */}
                {Object.entries(transcriptData).length === 0 && (
                    <p className="text-gray-400 col-span-full text-center py-8">
                        等待发言...
                    </p>
                )}
                {/* 遍历每个发言人的转录数据并渲染 */}
                {Object.entries(transcriptData).map(([speakerId, transcripts]) => (
                    <div
                        key={speakerId} // 添加 key 属性
                        className={`bg-gray-900 p-3 rounded-lg shadow-inner border transition-all duration-300 
                                             ${activeSpeakers.has(speakerId) ? 'border-blue-500 ring-2 ring-blue-500' : 'border-gray-700'}`}
                    >
                        <h3 className="font-bold text-lg text-blue-200 mb-2 flex items-center">
                            <User className="w-5 h-5 mr-2 text-blue-400" />
                            {speakerId} {/* 显示发言人 ID */}
                            {/* 如果是活跃发言人，显示“正在发言”标记 */}
                            {activeSpeakers.has(speakerId) && (
                                <span className="ml-2 text-xs bg-green-500 text-white px-2 py-0.5 rounded-full animate-pulse">正在发言</span>
                            )}
                        </h3>
                        {/* 转录文本显示区域，带有自定义滚动条 */}
                        <div
                            ref={el => scrollRefs.current[speakerId] = el} // 绑定 ref
                            className="text-gray-300 text-sm h-32 overflow-y-auto custom-scrollbar pr-2"
                        >
                            {/* 遍历并显示每个转录条目 */}
                            {transcripts.map((entry, index) => (
                                <p key={index} className={`mb-1 ${entry.isFinal ? 'text-gray-100' : 'text-gray-400 italic'}`}>
                                    {/* 显示时间戳 */}
                                    <span className="text-gray-500 text-xs mr-1">[{new Date(entry.timestamp).toLocaleTimeString()}]</span>
                                    {entry.text} {/* 显示转录文本 */}
                                </p>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// --- 用户列表组件 ---
function UserList() {
    const { users, activeSpeakers, systemStatus } = useContext(AppContext);
    const [onlineUsers, setOnlineUsers] = useState([]);
    const [showAllUsers, setShowAllUsers] = useState(false);
    const [allUsers, setAllUsers] = useState([]);
    const [editingUser, setEditingUser] = useState(null);
    const [editForm, setEditForm] = useState({
        voiceprint_id: '',
        new_role: '',
        new_name: ''
    });

    const UserRole = {
        GUEST: 'GUEST',
        MEMBER: 'MEMBER',
        ADMIN: 'ADMIN',
        HOST: 'HOST',
        MANAGER: 'MANAGER',
        CLIENT: 'CLIENT',
        REGISTERED_USER: 'REGISTERED_USER'
    };

    // 获取所有用户
    const fetchAllUsers = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/get-all-users`);
            if (!response.ok) throw new Error('获取用户失败');
            const data = await response.json();
            setAllUsers(data);
        } catch (error) {
            console.error('获取全部用户失败:', error);
            alert('获取用户列表失败，请检查网络连接');
        }
    };

    // 增强版在线用户过滤
    useEffect(() => {
        const now = Date.now();
        const threeHours = 3 * 60 * 60 * 1000;

        // 1. 获取系统当前连接的用户ID
        const conns = systemStatus?.connections_active || [];
        const connectedUserIds = Array.isArray(conns)
            ? conns.map(c => c.userId || c._id).filter(Boolean)
            : [];

        // 2. 获取3小时内活跃用户
        const recentlyActiveUsers = users.filter(user => {
            const lastActiveTime = new Date(user.last_active).getTime();
            return now - lastActiveTime <= threeHours;
        });

        // 3. 合并三类用户：
        //    - 当前连接的
        //    - 3小时内活跃的 
        //    - 正在发言的
        const combinedUsers = users.filter(user =>
            connectedUserIds.includes(user._id) ||
            recentlyActiveUsers.some(u => u._id === user._id) ||
            activeSpeakers.has(user._id)
        );

        // 去重后设置状态
        setOnlineUsers([...new Map(combinedUsers.map(user => [user._id, user])).values()]);
    }, [users, activeSpeakers, systemStatus]);

    useEffect(() => {
        const handleMeetingInit = (event) => {
            if (event.detail?.type === 'meeting_init_response') {
                const { userId, clientId } = event.detail;
                console.log('新用户加入会议:', userId || clientId);

                // 强制刷新在线用户列表
                setOnlineUsers(prev => {
                    const newUser = users.find(u => u._id === (userId || clientId));
                    if (newUser && !prev.some(u => u._id === newUser._id)) {
                        return [...prev, newUser];
                    }
                    return prev;
                });
            }
        };

        window.addEventListener('meeting_event', handleMeetingInit);
        return () => window.removeEventListener('meeting_event', handleMeetingInit);
    }, [users]);

    // 处理编辑用户
    const handleEditUser = (user) => {
        setEditingUser(user);
        setEditForm({
            voiceprint_id: user._id,
            new_role: user.role,
            new_name: user.name
        });
    };

    // 提交编辑
    const handleEditSubmit = async () => {
        if (!editingUser) return;

        try {
            const params = new URLSearchParams();
            params.append('voiceprint_id', editForm.voiceprint_id);
            params.append('new_role', editForm.new_role);

            if (editForm.new_name && editForm.new_name.trim() !== '') {
                params.append('new_name', editForm.new_name);
            }

            const response = await fetch(`${API_BASE_URL}/update-role?${params.toString()}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '更新用户失败');
            }

            const updatedUser = {
                ...editingUser,
                role: editForm.new_role,
                ...(editForm.new_name && { name: editForm.new_name })
            };

            const updateUsers = (userList) =>
                userList.map(u => u._id === editingUser._id ? updatedUser : u);

            setAllUsers(updateUsers(allUsers));
            setOnlineUsers(updateUsers(onlineUsers));

            setEditingUser(null);
            alert('用户信息更新成功');
        } catch (error) {
            console.error('更新用户失败:', error);
            alert(`更新失败: ${error.message}`);
        }
    };

    // 计算用户最后活跃时间描述
    const getLastActiveText = (lastActive) => {
        const now = new Date();
        const lastActiveTime = new Date(lastActive);
        const diffInMinutes = Math.floor((now - lastActiveTime) / (1000 * 60));

        if (diffInMinutes < 1) return '刚刚活跃';
        if (diffInMinutes < 60) return `${diffInMinutes}分钟前`;
        if (diffInMinutes < 24 * 60) return `${Math.floor(diffInMinutes / 60)}小时前`;
        return `${Math.floor(diffInMinutes / (60 * 24))}天前`;
    };

    return (
        <div className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700">
            <div className="flex justify-between items-center mb-2">
                <h2 className="text-xl font-semibold text-blue-300 flex items-center">
                    <Users className="w-6 h-6 mr-2 text-blue-400" />
                    在线成员 ({onlineUsers.length}/{users.length})
                </h2>
                <button
                    onClick={() => {
                        fetchAllUsers();
                        setShowAllUsers(true);
                    }}
                    className="text-gray-400 hover:text-blue-300 transition-colors p-1 rounded-full hover:bg-gray-700"
                    title="显示全部用户"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h.01M12 12h.01M19 12h.01" />
                    </svg>
                </button>
            </div>

            {/* 在线用户列表 */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 max-h-64 overflow-y-auto custom-scrollbar pr-2">
                {onlineUsers.length === 0 ? (
                    <p className="text-gray-400 col-span-full text-center py-4">
                        当前没有在线成员
                    </p>
                ) : (
                    onlineUsers.map((user) => (
                        <div
                            key={user._id}
                            className={`bg-gray-900 p-3 rounded-lg shadow-inner border transition-all duration-300 flex flex-col items-center justify-between
                                        ${activeSpeakers.has(user._id) ? 'border-blue-500 ring-2 ring-blue-500' : 'border-gray-700'}`}
                        >
                            <User className={`w-10 h-10 mb-2 ${activeSpeakers.has(user._id) ? 'text-blue-400' : 'text-gray-300'}`} />
                            <span className="text-sm font-medium text-gray-100 text-center truncate w-full">
                                {user.name}
                                {activeSpeakers.has(user._id) && (
                                    <span className="ml-1 text-xs text-green-400">• 发言中</span>
                                )}
                            </span>
                            <span className="text-xs text-gray-400 mt-1 px-2 py-0.5 rounded bg-gray-700">
                                {user.role}
                            </span>
                            <div className="text-xs text-gray-500 mt-1">
                                {getLastActiveText(user.last_active)}
                            </div>
                            <button
                                onClick={() => handleEditUser(user)}
                                className="mt-2 text-blue-400 hover:text-blue-300 transition-colors duration-200 text-xs flex items-center"
                            >
                                <Edit className="w-3 h-3 mr-1" /> 编辑
                            </button>
                        </div>
                    ))
                )}
            </div>

            {/* 全部用户弹窗 */}
            {showAllUsers && (
                <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
                    <div className="bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700 w-full max-w-2xl max-h-[80vh] flex flex-col">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl font-bold text-blue-300">
                                全部用户 ({allUsers.length})
                                <span className="text-sm text-green-400 ml-2">
                                    在线: {onlineUsers.length}
                                </span>
                            </h3>
                            <button
                                onClick={() => setShowAllUsers(false)}
                                className="text-gray-400 hover:text-white transition-colors"
                            >
                                <XCircle className="w-6 h-6" />
                            </button>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 overflow-y-auto custom-scrollbar pr-2">
                            {allUsers.map(user => (
                                <div
                                    key={user._id}
                                    className={`bg-gray-900 p-3 rounded-lg border ${onlineUsers.some(u => u._id === user._id)
                                        ? 'border-green-500'
                                        : 'border-gray-700'
                                        }`}
                                >
                                    <div className="flex items-center space-x-2">
                                        <User className={`w-5 h-5 ${onlineUsers.some(u => u._id === user._id)
                                            ? 'text-green-400'
                                            : 'text-gray-500'
                                            }`} />
                                        <span className="text-sm font-medium text-gray-100 truncate">
                                            {user.name}
                                        </span>
                                    </div>
                                    <div className="mt-1 text-xs text-gray-400">
                                        <div>角色: {user.role}</div>
                                        <div>ID: {user._id.slice(0, 6)}...</div>
                                        <div>最后活跃: {new Date(user.last_active).toLocaleString()}</div>
                                    </div>
                                    <button
                                        onClick={() => handleEditUser(user)}
                                        className="mt-2 w-full text-xs text-blue-400 hover:text-blue-300 flex items-center justify-center"
                                    >
                                        <Edit className="w-3 h-3 mr-1" /> 编辑
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* 编辑用户模态框 */}
            {editingUser && (
                <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
                    <div className="bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700 w-full max-w-md">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl font-bold text-blue-300">编辑用户</h3>
                            <button
                                onClick={() => setEditingUser(null)}
                                className="text-gray-400 hover:text-white transition-colors"
                            >
                                <XCircle className="w-6 h-6" />
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-300 mb-1">用户ID</label>
                                <input
                                    type="text"
                                    value={editForm.voiceprint_id}
                                    readOnly
                                    className="w-full p-2 rounded bg-gray-700 border border-gray-600 text-gray-400 cursor-not-allowed"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-300 mb-1">用户名</label>
                                <input
                                    type="text"
                                    value={editForm.new_name}
                                    onChange={(e) => setEditForm({
                                        ...editForm,
                                        new_name: e.target.value
                                    })}
                                    className="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-300 mb-1">角色</label>
                                <select
                                    value={editForm.new_role}
                                    onChange={(e) => setEditForm({
                                        ...editForm,
                                        new_role: e.target.value
                                    })}
                                    className="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white"
                                >
                                    {Object.entries(UserRole).map(([key, value]) => (
                                        <option key={value} value={value}>
                                            {key} ({value})
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="flex justify-end space-x-2 mt-4">
                                <button
                                    onClick={() => setEditingUser(null)}
                                    className="px-4 py-2 bg-gray-700 text-gray-200 rounded hover:bg-gray-600"
                                >
                                    取消
                                </button>
                                <button
                                    onClick={handleEditSubmit}
                                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                                >
                                    保存更改
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// --- 发言人编辑模态框组件 ---
function SpeakerEditModal({ user, onClose, onSave }) {
    // 状态：新用户名和新角色
    const [newUsername, setNewUsername] = useState(user.name);
    const [newRole, setNewRole] = useState(user.role);

    // 可选的角色列表
    const roles = ["GUEST", "MEMBER", "ADMIN", "HOST", "CLIENT", "REGISTERED_USER", "UNKNOWN", "ERROR"];

    // 处理保存按钮点击事件
    const handleSave = () => {
        if (newUsername.trim() === "") {
            alert("用户名不能为空。");
            return;
        }
        onSave(user._id, newUsername, newRole); // 调用父组件传入的保存函数
    };

    return (
        // 固定定位的背景遮罩层
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
            {/* 模态框内容区域 */}
            <div className="bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700 w-full max-w-md animate-fade-in-up">
                <h3 className="text-2xl font-bold text-blue-300 mb-5">编辑发言人信息</h3>
                {/* 用户名输入框 */}
                <div className="mb-4">
                    <label htmlFor="name" className="block text-gray-300 text-sm font-medium mb-2">用户名:</label>
                    <input
                        type="text"
                        id="name"
                        value={newUsername}
                        onChange={(e) => setNewUsername(e.target.value)}
                        className="w-full p-3 rounded-lg bg-gray-900 border border-gray-700 text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    />
                </div>
                {/* 角色选择下拉框 */}
                <div className="mb-6">
                    <label htmlFor="role" className="block text-gray-300 text-sm font-medium mb-2">角色:</label>
                    <select
                        id="role"
                        value={newRole}
                        onChange={(e) => setNewRole(e.target.value)}
                        className="w-full p-3 rounded-lg bg-gray-900 border border-gray-700 text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    >
                        {roles.map(roleOption => (
                            <option key={roleOption} value={roleOption}> {/* 确保 option 也有 key */}
                                {roleOption}
                            </option>
                        ))}
                    </select>
                </div>
                {/* 底部操作按钮 */}
                <div className="flex justify-end space-x-3">
                    <button
                        onClick={onClose}
                        className="px-5 py-2 rounded-lg bg-gray-700 text-gray-200 hover:bg-gray-600 transition-colors duration-200 shadow-md"
                    >
                        取消
                    </button>
                    <button
                        onClick={handleSave}
                        className="px-5 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors duration-200 shadow-md"
                    >
                        保存
                    </button>
                </div>
            </div>
        </div>
    );
}

// --- 控制面板组件 ---
function ControlPanel() {
    // 从上下文中获取处理函数
    const { handleGenerateMinutes, handleGenerateReport, handleExportUserSpeech, handleLinkMeetingToKG } = useContext(AppContext);

    return (
        <div className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700">
            <h2 className="text-xl font-semibold text-blue-300 mb-3">操作面板</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* 抽取知识图谱按钮 */}
                <button
                    onClick={handleLinkMeetingToKG}
                    className="flex items-center justify-center px-4 py-3 bg-purple-600 text-white rounded-lg shadow-md hover:bg-purple-700 transition-all duration-200 text-sm font-medium"
                >
                    <Book className="w-5 h-5 mr-2" />
                    抽取三元图谱
                </button>
                {/* 导出内容按钮 (带下拉菜单) */}
                <div className="relative group">
                    <button
                        className="flex items-center justify-center w-full px-4 py-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 transition-all duration-200 text-sm font-medium"
                    >
                        <Download className="w-5 h-5 mr-2" />
                        导出内容
                    </button>
                    {/* 下拉菜单内容 */}
                    <div className="absolute right-0 mt-2 w-48 bg-gray-700 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform scale-95 group-hover:scale-100 z-10">
                        <button
                            onClick={handleGenerateMinutes}
                            className="block w-full text-left px-4 py-2 text-gray-200 hover:bg-gray-600 rounded-t-lg text-sm"
                        >
                            导出会议纪要 (.docx)
                        </button>
                        <button
                            onClick={handleGenerateReport}
                            className="block w-full text-left px-4 py-2 text-gray-200 hover:bg-gray-600 text-sm"
                        >
                            生成会议总结报告
                        </button>
                        <button
                            onClick={() => handleExportUserSpeech(null)} // 导出所有发言
                            className="block w-full text-left px-4 py-2 text-gray-200 hover:bg-gray-600 rounded-b-lg text-sm"
                        >
                            导出所有发言 (.txt)
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

// --- LLM 问答窗口组件 ---
function LLMQAWindow() {
    // 从上下文中获取 LLM 问答历史、加载状态和提问函数
    const { llmQAHistory, llmQALoading, handleLLMQuestion } = useContext(AppContext);
    const [question, setQuestion] = useState('');
    const chatContainerRef = useRef(null);

    // 处理表单提交 (用户提问)
    const handleSubmit = async (e) => {
        e.preventDefault(); // 阻止表单默认提交行为
        if (question.trim() === '' || llmQALoading) return; // 如果问题为空或正在加载，则不执行
        await handleLLMQuestion(question); // 调用 LLM 问答函数
        setQuestion(''); // 清空输入框
    };

    // useEffect 在 LLM 问答历史变化时滚动到最新消息
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight; // 滚动到底部
        }
    }, [llmQAHistory]); // 依赖项：当 llmQAHistory 变化时执行

    return (
        <div className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 flex flex-col h-96">
            <h2 className="text-xl font-semibold text-blue-300 mb-3 flex items-center">
                <MessageSquare className="w-6 h-6 mr-2 text-blue-400" />
                RAG
            </h2>
            {/* 聊天消息显示区域 */}
            <div ref={chatContainerRef} className="flex-1 overflow-y-auto custom-scrollbar p-2 mb-4 bg-gray-900 rounded-lg border border-gray-700">
                {/* 如果没有消息，显示提示信息 */}
                {llmQAHistory.length === 0 && (
                    <p className="text-gray-400 text-center py-8">
                        向 RAG 提问会议内容...
                    </p>
                )}
                {/* 遍历并显示每条消息 */}
                {llmQAHistory.map((msg, index) => (
                    <div key={index} className={`mb-2 p-2 rounded-lg ${msg.type === 'user' ? 'bg-blue-700 text-white ml-auto max-w-[80%]' : 'bg-gray-700 text-gray-100 mr-auto max-w-[80%]'}`}>
                        <p className="font-semibold text-xs mb-1">{msg.type === 'user' ? '您' : 'RAG'}</p>
                        <p className="whitespace-pre-wrap text-sm">{msg.text}</p>
                    </div>
                ))}
                {/* 加载指示器 */}
                {llmQALoading && (
                    <div className="flex items-center justify-center p-2 text-gray-400">
                        <Loader className="w-5 h-5 animate-spin mr-2" />
                        思考中...
                    </div>
                )}
            </div>
            {/* 提问输入框和发送按钮 */}
            <form onSubmit={handleSubmit} className="flex space-x-2">
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="输入您的问题..."
                    className="flex-1 p-3 rounded-lg bg-gray-900 border border-gray-700 text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    disabled={llmQALoading} // 加载时禁用输入
                />
                <button
                    type="submit"
                    className="p-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={llmQALoading || question.trim() === ''} // 加载或问题为空时禁用按钮
                >
                    <Send className="w-5 h-5" />
                </button>
            </form>
        </div>
    );
}

// --- 系统状态显示组件 ---
function SystemStatusDisplay() {
    // 从上下文中获取系统状态
    const { systemStatus } = useContext(AppContext);

    // 根据加载状态返回不同的图标
    const getStatusIcon = (isLoaded) => {
        return isLoaded ? <CheckCircle className="w-4 h-4 text-green-500 mr-1" /> : <XCircle className="w-4 h-4 text-red-500 mr-1" />;
    };

    return (
        <div className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700">
            <h2 className="text-xl font-semibold text-blue-300 mb-3 flex items-center">
                <Settings className="w-6 h-6 mr-2 text-blue-400" />
                系统状态
            </h2>
            <div className="space-y-2 text-sm text-gray-300">
                {/* 各项系统指标显示 */}
                <p className="flex items-center">
                    <span className="font-medium w-32">总连接数:</span>
                    <span className="text-blue-200">{systemStatus.total_connections || 0}</span>
                </p>
                <p className="flex items-center">
                    <span className="font-medium w-32">活跃连接数:</span>
                    <span className="text-blue-200">{systemStatus.connections_active || 0}</span>
                </p>
                <p className="flex items-center">
                    <span className="font-medium w-32">活跃会议数:</span>
                    <span className="text-blue-200">{systemStatus.meetings_active || 0}</span>
                </p>
                <p className="flex items-center">
                    <span className="font-medium w-32">音频块处理数:</span>
                    <span className="text-blue-200">{systemStatus.audio_processed || 0}</span>
                </p>
                {/* 模型加载状态 */}
                <div className="border-t border-gray-700 pt-2 mt-2">
                    <p className="flex items-center">
                        {getStatusIcon(systemStatus.stt_model_loaded)}
                        <span className="font-medium">VAD 模型:</span>
                        <span className="ml-2">{systemStatus.stt_model_loaded ? '已加载' : '未加载'}</span>
                    </p>
                    <p className="flex items-center">
                        {getStatusIcon(systemStatus.stt_model_loaded)}
                        <span className="font-medium">STT 模型:</span>
                        <span className="ml-2">{systemStatus.stt_model_loaded ? '已加载' : '未加载'}</span>
                    </p>
                    <p className="flex items-center">
                        {getStatusIcon(systemStatus.speaker_model_loaded)}
                        <span className="font-medium">声纹模型:</span>
                        <span className="ml-2">{systemStatus.speaker_model_loaded ? '已加载' : '未加载'}</span>
                    </p>
                    <p className="flex items-center">
                        {getStatusIcon(systemStatus.summary_model_loaded)}
                        <span className="font-medium">摘要模型:</span>
                        <span className="ml-2">{systemStatus.summary_model_loaded ? '已加载' : '未加载'}</span>
                    </p>
                    <p className="flex items-center">
                        {getStatusIcon(systemStatus.llm_model_loaded)}
                        <span className="font-medium">LLM 模型:</span>
                        <span className="ml-2">{systemStatus.llm_model_loaded ? '已加载' : '未加载'}</span>
                    </p>
                    <p className="flex items-center">
                        {getStatusIcon(systemStatus.bge_model_loaded)}
                        <span className="font-medium">BGE 模型:</span>
                        <span className="ml-2">{systemStatus.bge_model_loaded ? '已加载' : '未加载'}</span>
                    </p>
                </div>
            </div>
        </div>
    );
}

// 导出主应用组件
export default App;


import React, { useState, useCallback, useEffect } from 'react';
import { Mic, Loader, CheckCircle, XCircle, Shield } from 'lucide-react';

// 后端 API 基础 URL (与 App.jsx 保持一致)
// 后端 API 基础 URL (与 App.jsx 保持一致)
const API_BASE_URL = import.meta.env.PROD
    ? window.location.origin
    : (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000');

// 声纹登录/注册组件
function VoiceLogin({ onLoginSuccess }) {
    const [textToRead, setTextToRead] = useState('');
    const [status, setStatus] = useState('initial'); // 'initial', 'fetching_text', 'ready_to_record', 'recording', 'verifying', 'success', 'error'
    const [errorMessage, setErrorMessage] = useState('');
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [audioChunks, setAudioChunks] = useState([]);

    // 1. 获取需要朗读的文本
    const fetchTextToRead = useCallback(async () => {
        setStatus('fetching_text');
        setErrorMessage('');
        try {
            const response = await fetch(`${API_BASE_URL}/get_voice_login_text`);
            if (!response.ok) {
                throw new Error('获取登录文本失败');
            }
            const data = await response.json();
            setTextToRead(data.text);
            setStatus('ready_to_record');
        } catch (error) {
            console.error(error);
            setErrorMessage('无法连接到服务器，请稍后重试。');
            setStatus('error');
        }
    }, []);

    // 组件加载时自动获取文本
    useEffect(() => {
        fetchTextToRead();
    }, [fetchTextToRead]);

    // 2. 开始录音
    const startRecording = async () => {
        if (status !== 'ready_to_record') return;

        setAudioChunks([]);
        setStatus('recording');
        setErrorMessage('');

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            setMediaRecorder(recorder);

            recorder.ondataavailable = (event) => {
                setAudioChunks(prev => [...prev, event.data]);
            };

            recorder.onstop = () => {
                stream.getTracks().forEach(track => track.stop());
                // onstop 事件触发后，audioChunks 可能还未完全更新，
                // 所以我们将处理逻辑移到这里，并使用一个回调来获取最新的 chunks
                setAudioChunks(currentChunks => {
                    handleVoiceSubmission(currentChunks);
                    return currentChunks; // 返回当前 chunks 以更新状态
                });
            };

            recorder.start();

            // 录制5秒后自动停止
            setTimeout(() => {
                if (recorder.state === 'recording') {
                    recorder.stop();
                }
            }, 5000);

        } catch (error) {
            console.error('麦克风访问失败:', error);
            setErrorMessage('无法访问麦克风。请检查浏览器权限。');
            setStatus('error');
        }
    };

    // 3. 提交音频进行验证/注册
    const handleVoiceSubmission = async (chunks) => {
        if (chunks.length === 0) {
            setErrorMessage('录音失败，未捕获到音频。');
            setStatus('error');
            return;
        }

        setStatus('verifying');
        const audioBlob = new Blob(chunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'voice_login.wav');

        try {
            const response = await fetch(`${API_BASE_URL}/voice_login`, {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || '声纹登录失败');
            }

            setStatus('success');
            // 登录成功后，调用父组件的回调函数，并传递用户信息
            onLoginSuccess(result.user);

        } catch (error) {
            console.error('声纹登录/注册失败:', error);
            setErrorMessage(error.message || '验证失败，请重试。');
            setStatus('error');
        }
    };

    // 根据状态渲染不同的UI
    const renderStatus = () => {
        switch (status) {
            case 'recording':
                return <><Loader className="w-5 h-5 mr-2 animate-spin" /> 正在录音 (5秒)...</>;
            case 'verifying':
                return <><Loader className="w-5 h-5 mr-2 animate-spin" /> 正在验证声纹...</>;
            case 'success':
                return <><CheckCircle className="w-5 h-5 mr-2 text-green-400" /> 验证成功！</>;
            case 'error':
                return <><XCircle className="w-5 h-5 mr-2 text-red-400" /> {errorMessage}</>;
            case 'fetching_text':
                return <><Loader className="w-5 h-5 mr-2 animate-spin" /> 正在获取指令...</>;
            default:
                return '请按住按钮并朗读以下文本';
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center p-4">
            <div className="w-full max-w-md text-center bg-gray-800 p-8 rounded-2xl shadow-2xl border border-gray-700">
                <Shield className="w-16 h-16 mx-auto text-blue-400 mb-4" />
                <h1 className="text-3xl font-bold text-gray-100 mb-2">声纹登录</h1>
                <p className="text-gray-400 mb-6">{renderStatus()}</p>

                <div className="bg-gray-900 p-6 rounded-lg mb-8 border border-gray-700 min-h-[100px] flex items-center justify-center">
                    <p className="text-2xl font-mono text-blue-300 tracking-wider">
                        {status === 'fetching_text' ? '...' : textToRead}
                    </p>
                </div>

                <button
                    onClick={startRecording}
                    disabled={status !== 'ready_to_record' && status !== 'error'}
                    className={`w-24 h-24 rounded-full mx-auto flex items-center justify-center transition-all duration-300 ease-in-out
                                ${status === 'recording' ? 'bg-red-600 animate-pulse' : 'bg-blue-600 hover:bg-blue-700'}
                                disabled:bg-gray-600 disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-blue-500/50`}
                >
                    <Mic className="w-10 h-10 text-white" />
                </button>

                {status === 'error' && (
                    <button
                        onClick={fetchTextToRead}
                        className="mt-6 text-sm text-blue-400 hover:underline"
                    >
                        重试
                    </button>
                )}
            </div>
        </div>
    );
}

export default VoiceLogin;

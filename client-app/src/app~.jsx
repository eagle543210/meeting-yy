//导入并组合所有子组件，并负责管理 WebSocket 连接、状态管理以及应用程序的整体布局。
import React, { useState, createContext, useEffect, useRef } from 'react';
import { Mic, Pause, StopCircle, Users, Settings, MessageSquare, Sun, Moon } from 'lucide-react';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import Chatbot from './components/Chatbot';
import SystemStatusDisplay from './components/SystemStatusDisplay';

export const AppContext = createContext();

function App() {
    const [transcription, setTranscription] = useState([]);
    const [systemStatus, setSystemStatus] = useState({});
    const [audioLevel, setAudioLevel] = useState(0); // 新增：音量状态
    const wsRef = useRef(null);
    const [isListening, setIsListening] = useState(false);
    const [isDarkMode, setIsDarkMode] = useState(true);

    useEffect(() => {
        if (!wsRef.current) {
            wsRef.current = new WebSocket(import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000/ws');

            wsRef.current.onopen = () => {
                console.log('WebSocket 连接成功');
                wsRef.current.send(JSON.stringify({ type: "start_listening" }));
                setIsListening(true);
            };

            wsRef.current.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'transcription_update') {
                    setTranscription(message.payload);
                } else if (message.type === 'status_update') {
                    setSystemStatus(message.payload);
                } else if (message.type === 'audio_level') { // 新增：处理音量消息
                    setAudioLevel(message.level);
                }
            };

            wsRef.current.onclose = () => {
                console.log('WebSocket 连接断开');
                setIsListening(false);
                setSystemStatus({});
            };

            wsRef.current.onerror = (error) => {
                console.error('WebSocket 错误:', error);
                setIsListening(false);
            };
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    const handleStartStop = () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const action = isListening ? "stop_listening" : "start_listening";
            wsRef.current.send(JSON.stringify({ type: action }));
            setIsListening(!isListening);
        } else {
            console.error('WebSocket 未连接，无法发送指令。');
        }
    };
    
    return (
        <AppContext.Provider value={{ systemStatus, transcription }}>
            <div className={`min-h-screen ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-white text-gray-900'} transition-colors duration-300`}>
                <header className={`p-4 ${isDarkMode ? 'bg-gray-800 border-b border-gray-700' : 'bg-gray-100 border-b border-gray-200'} shadow-md flex justify-between items-center`}>
                    <h1 className="text-2xl font-bold flex items-center">
                        <Mic className="w-7 h-7 mr-2 text-blue-500" />
                        AI 会议助手
                    </h1>
                    <div className="flex items-center space-x-4">
                        <button
                            onClick={handleStartStop}
                            className={`flex items-center p-2 rounded-full shadow-md ${isListening ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'} text-white transition-colors duration-200`}
                        >
                            {isListening ? (
                                <><StopCircle className="w-5 h-5 mr-2" /> 停止</>
                            ) : (
                                <><Mic className="w-5 h-5 mr-2" /> 开始</>
                            )}
                        </button>
                        <button onClick={() => setIsDarkMode(!isDarkMode)} className={`p-2 rounded-full transition-colors duration-200 ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-300 hover:bg-gray-400'}`}>
                            {isDarkMode ? <Moon className="w-5 h-5 text-yellow-300" /> : <Sun className="w-5 h-5 text-orange-500" />}
                        </button>
                    </div>
                </header>
                <main className="flex-1 p-4 flex space-x-4 overflow-hidden">
                    <div className="w-1/3 flex flex-col space-y-4">
                        {/* 传递 audioLevel 状态给 SystemStatusDisplay */}
                        <SystemStatusDisplay systemStatus={systemStatus} audioLevel={audioLevel} />
                    </div>
                    <div className="flex-1 flex flex-col space-y-4">
                        <div className="flex-1">
                            <TranscriptionDisplay transcription={transcription} />
                        </div>
                        <div className="flex-1">
                            <Chatbot />
                        </div>
                    </div>
                </main>
            </div>
        </AppContext.Provider>
    );
}

export default App;
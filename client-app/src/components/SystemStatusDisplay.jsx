//这个组件只负责显示系统状态信息，使其逻辑与主应用分离。
import React from 'react';
import { CheckCircle, XCircle, Settings, Volume2 } from 'lucide-react';

function SystemStatusDisplay({ systemStatus, audioLevel }) {
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
                
                {/* 完整的音量强度显示窗口 */}
                <div className="border-t border-gray-700 pt-2 mt-2">
                    <p className="flex items-center text-sm text-gray-300 font-medium mb-1">
                        <Volume2 className="w-4 h-4 mr-2" />
                        音量强度:
                    </p>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                        <div 
                            className="bg-blue-500 h-2 rounded-full transition-all duration-100" 
                            style={{ width: `${audioLevel * 100}%` }}
                        ></div>
                    </div>
                </div>

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

export default SystemStatusDisplay;
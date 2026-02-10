//这个组件负责渲染会议转录文本，并根据发言人状态进行高亮。
import React from 'react';
import { Mic } from 'lucide-react';

function TranscriptionDisplay({ transcription }) {
    return (
        <div className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 h-96 overflow-y-auto custom-scrollbar">
            <h2 className="text-xl font-semibold text-blue-300 mb-3 flex items-center">
                <Mic className="w-6 h-6 mr-2 text-blue-400" />
                会议转录
            </h2>
            <div className="space-y-4">
                {transcription.length === 0 ? (
                    <p className="text-gray-400 text-center py-8">
                        等待会议开始...
                    </p>
                ) : (
                    transcription.map((item, index) => (
                        <div key={index} className={`flex items-start ${item.is_current ? 'font-bold text-white' : 'text-gray-300'}`}>
                            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm mr-2 ${item.is_current ? 'bg-blue-500' : 'bg-gray-600'}`}>
                                {item.speaker_name[0]}
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="font-semibold text-sm mb-1">{item.speaker_name}</p>
                                <p className="whitespace-pre-wrap text-sm">{item.text}</p>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

export default TranscriptionDisplay;
//这个组件专注于 RAG 聊天机器人功能，包括消息显示和用户输入。
import React, { useState, useRef, useEffect } from 'react';
import { Loader, Send } from 'lucide-react';

function Chatbot() {
    const [question, setQuestion] = useState('');
    const [llmQAHistory, setLlmQAHistory] = useState([]);
    const [llmQALoading, setLlmQALoading] = useState(false);
    const chatContainerRef = useRef(null);

    const handleAskQuestion = async () => {
        if (question.trim() === '') return;

        const userQuestion = question;
        setLlmQAHistory(prev => [...prev, { type: 'user', text: userQuestion }]);
        setQuestion('');
        setLlmQALoading(true);

        try {
            // 这里是调用后端 RAG API 的逻辑，为了示例，我们使用模拟延迟
            await new Promise(resolve => setTimeout(resolve, 1500));
            const dummyResponse = "这是一个来自 RAG 系统的回答。";

            setLlmQAHistory(prev => [...prev, { type: 'rag', text: dummyResponse }]);
        } catch (error) {
            console.error('RAG 查询失败:', error);
            setLlmQAHistory(prev => [...prev, { type: 'rag', text: '对不起，查询失败。' }]);
        } finally {
            setLlmQALoading(false);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        handleAskQuestion();
    };

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [llmQAHistory]);

    return (
        <div className="flex-1 overflow-hidden flex flex-col p-4 bg-gray-900 rounded-lg border border-gray-700">
            <div ref={chatContainerRef} className="flex-1 overflow-y-auto custom-scrollbar p-2 mb-4">
                {llmQAHistory.length === 0 && (
                    <p className="text-gray-400 text-center py-8">
                        向 RAG 提问会议内容...
                    </p>
                )}
                {llmQAHistory.map((msg, index) => (
                    <div key={index} className={`mb-2 p-2 rounded-lg ${msg.type === 'user' ? 'bg-blue-700 text-white ml-auto max-w-[80%]' : 'bg-gray-700 text-gray-100 mr-auto max-w-[80%]'}`}>
                        <p className="font-semibold text-xs mb-1">{msg.type === 'user' ? '您' : 'RAG'}</p>
                        <p className="whitespace-pre-wrap text-sm">{msg.text}</p>
                    </div>
                ))}
                {llmQALoading && (
                    <div className="flex items-center justify-center p-2 text-gray-400">
                        <Loader className="w-5 h-5 animate-spin mr-2" />
                        思考中...
                    </div>
                )}
            </div>
            <form onSubmit={handleSubmit} className="flex space-x-2">
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="输入您的问题..."
                    className="flex-1 p-3 rounded-lg bg-gray-900 border border-gray-700 text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    disabled={llmQALoading}
                />
                <button
                    type="submit"
                    className="p-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={llmQALoading || question.trim() === ''}
                >
                    <Send className="w-5 h-5" />
                </button>
            </form>
        </div>
    );
}

export default Chatbot;
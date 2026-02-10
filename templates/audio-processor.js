// static/js/audio-processor.js

// 添加工作线程注册检查
if (typeof AudioWorkletProcessor !== 'undefined') {
    class AudioProcessor extends AudioWorkletProcessor {
        process(inputs) {
            const input = inputs[0];
            if (input && input.length > 0) {
                const audioData = new Int16Array(input[0].length);
                for (let i = 0; i < input[0].length; i++) {
                    audioData[i] = Math.min(32767, Math.max(-32768, input[0][i] * 32768));
                }
                this.port.postMessage({ 
                    audioBuffer: Array.from(audioData) 
                });
            }
            return true;
        }
    }
    
    registerProcessor('audio-processor', AudioProcessor);
} else {
    console.warn('AudioWorkletProcessor未定义，当前环境不支持Worklet');
}
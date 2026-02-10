// static/js/audio-processor.js

class AudioProcessor extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const inputChannel = input[0];
            const audioData = new Float32Array(inputChannel.length);
            
            // 计算音量
            let sum = 0;
            for (let i = 0; i < inputChannel.length; i++) {
                audioData[i] = inputChannel[i];
                sum += inputChannel[i] * inputChannel[i];
            }
            const rms = Math.sqrt(sum / inputChannel.length);
            const volume = Math.min(100, Math.round(rms * 1000));
            
            // 发送音频数据和音量
            this.port.postMessage({
                audioData: Array.from(audioData),
                volumeLevel: volume
            });
        }
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);

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
// audio-processor.js

// 定义一个函数用于重采样音频数据
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
        // 使用简单的线性插值进行重采样
        const a = audioBuffer[index];
        const b = audioBuffer[Math.min(audioBuffer.length - 1, index + offset)];
        const fraction = (i / ratio) - index;
        result[i] = a + fraction * (b - a);
    }
    return result;
};

// AudioWorklet 处理器类
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.originalSampleRate = 0;
        this.targetSampleRate = 16000; // 默认目标采样率
        
        // 监听来自主线程的消息，获取采样率
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

        const inputBuffer = input[0]; // 获取单声道音频数据 (Float32Array)

        let resampledData = inputBuffer;
        // 只有当原始采样率不等于目标采样率时才进行重采样
        if (this.originalSampleRate !== this.targetSampleRate) {
            resampledData = resampleAudio(inputBuffer, this.originalSampleRate, this.targetSampleRate);
        }

        // 将重采样后的 Float32 数据转换为 Int16 PCM 格式
        const pcm16 = new Int16Array(resampledData.length);
        for (let i = 0; i < resampledData.length; i++) {
            // 缩放到 16 位整数范围，并进行钳位处理
            pcm16[i] = Math.max(-1, Math.min(1, resampledData[i])) * 0x7FFF;
        }

        // 将处理后的数据发送回主线程
        this.port.postMessage({ type: 'audioData', pcm16Data: pcm16 });

        return true; // 保持处理器活跃
    }
}

// 注册 AudioWorklet 处理器
registerProcessor('audio-processor', AudioProcessor);

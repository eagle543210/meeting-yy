// M:\meeting\static\js\audio-processor.js

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = 16000; // 目标采样率，与后端匹配
        this.bufferSize = 4096; // 内部处理缓冲区大小，例如 4096 采样点
        this.audioBuffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;

        // 用于音量计算
        this.volume = 0;
        this.volumeAlpha = 0.9; // 平滑因子，用于平滑音量值
        // 初始音量发送，确保UI有初始值
        this.port.postMessage({ type: 'volume', volume: this.volume }); 
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0]; // 获取第一个输入流
        if (input.length === 0) {
            // 没有输入，可能麦克风已停止或静音
            this.port.postMessage({ type: 'volume', volume: 0 }); // 发送0音量
            return true;
        }

        const inputChannelData = input[0]; // 获取第一个通道的数据 (假设是单声道)

        // 计算音量 (RMS)
        let sum = 0;
        for (let i = 0; i < inputChannelData.length; i++) {
            sum += inputChannelData[i] * inputChannelData[i];
        }
        const rms = Math.sqrt(sum / inputChannelData.length);
        // 使用平滑的RMS，并确保音量不会突然降为0
        this.volume = Math.max(this.volume * this.volumeAlpha, rms); // 缓慢衰减音量，模拟VU表

        // 每次处理都发送音量更新，由主线程控制发送频率
        this.port.postMessage({ type: 'volume', volume: this.volume });


        // 将输入数据复制到内部缓冲区
        for (let i = 0; i < inputChannelData.length; i++) {
            if (this.bufferIndex < this.bufferSize) {
                this.audioBuffer[this.bufferIndex++] = inputChannelData[i];
            } else {
                // 缓冲区已满，发送数据并重置
                this.port.postMessage({ type: 'audioData', audioData: this.audioBuffer.buffer });
                // 将当前输入作为新缓冲区的开始
                // 确保不会溢出，只复制 bufferSize 长度的数据
                this.audioBuffer.set(inputChannelData.slice(0, this.bufferSize)); 
                this.bufferIndex = inputChannelData.length; // 更新索引
            }
        }

        // 如果缓冲区接近满或输入结束，发送剩余数据
        // 确保只在有足够数据时发送，避免发送空或不完整的数据
        if (this.bufferIndex >= this.bufferSize) { // 当缓冲区满时发送
            this.port.postMessage({ type: 'audioData', audioData: this.audioBuffer.buffer });
            this.bufferIndex = 0; // 重置索引
        }


        return true; // 保持处理器活跃
    }
}

registerProcessor('audio-processor', AudioProcessor);

// /static/js/audio_recorder.js

// 导入 AudioWorklet 加载器
// 注意: loadAudioProcessor 函数似乎不再直接使用，因为它在 AudioRecorder.start() 中直接加载了 AudioWorklet 模块
// 如果你的 audio_processor_loader.js 包含了更复杂的逻辑，可以保持导入。
// 但就目前代码来看，它不是必需的，因为 AudioWorklet 模块是直接在 start() 中 addModule 的。
// import { loadAudioProcessor } from './audio_processor_loader.js';

const SAMPLE_RATE = 16000; // 麦克风采样率，必须与后端VAD模型匹配
const CHUNK_SIZE = 512;    // 每个音频数据包的样本数，必须是VAD模型期望的输入尺寸

export class AudioRecorder {
    constructor(onVolumeUpdateCallback, wsClient) {
        this.audioContext = null;         // 音频上下文
        this.mediaStream = null;          // 媒体流（来自麦克风）
        this.microphone = null;           // 媒体流源节点 (与 mediaStreamSource 相同，这里统一用 microphone)
        this.audioProcessor = null;       // AudioWorkletNode 实例
        this.onVolumeUpdateCallback = onVolumeUpdateCallback; // 用于更新UI音量显示的回调函数
        this.wsClient = wsClient;         // WebSocketClient 实例

        this.isRecording = false;         // 录音状态标记
        this.audioQueue = [];             // 用于缓冲 AudioWorklet Processor 发送的音频数据
        this.sendInterval = 50;           // 每隔 50ms 发送一次数据，避免过于频繁
        this.sendTimer = null;            // 发送定时器 ID

    }

    async start() {
        if (this.isRecording) {
            console.warn("AudioRecorder: 录音已在进行中。");
            return;
        }

        try {
            //console.log("AudioRecorder: 尝试初始化 AudioContext...");
            // 使用 AudioContext 的构造函数传递采样率
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
            console.log("AudioRecorder: AudioContext 初始化成功，采样率:", this.audioContext.sampleRate); // 验证实际采样率
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
                console.log("AudioRecorder: AudioContext 已从 'suspended' 状态恢复。");
            }
            //console.log("AudioRecorder: 尝试加载 AudioWorklet 处理器...");
            // 直接在这里添加模块，而不是依赖外部 loader 函数
            await this.audioContext.audioWorklet.addModule('/static/js/audio-processor.js');
            //console.log("AudioRecorder: AudioWorklet 处理器加载成功。");

            //console.log("AudioRecorder: 尝试请求麦克风权限...");
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            //console.log("AudioRecorder: 麦克风权限已获取，媒体流获取成功。");

            this.microphone = this.audioContext.createMediaStreamSource(this.mediaStream);
            //console.log("AudioRecorder: 麦克风 MediaStreamSource 创建成功。");

            // 创建 AudioWorkletNode，并传递 chunkSize 给 processorOptions
            this.audioProcessor = new AudioWorkletNode(this.audioContext, 'audio-processor', {
                processorOptions: {
                    chunkSize: CHUNK_SIZE
                }
            });
            // console.log(`AudioRecorder: AudioProcessor (AudioWorkletNode) 创建成功，CHUNK_SIZE: ${CHUNK_SIZE}`);

            // 连接麦克风到 AudioWorkletNode
            this.microphone.connect(this.audioProcessor);
            // 将 AudioWorkletNode 连接到 AudioContext 的目的地，这是激活它的关键一步
            // 否则，它可能不会开始处理音频。但请注意，通常只需要输入节点连接到它。
            // 这里为了确保激活，暂时连接到目的地。
            this.audioProcessor.connect(this.audioContext.destination);
            console.log("AudioRecorder: 麦克风已连接到 AudioProcessor，并激活。");

            // 设置 AudioWorkletNode 的 port.onmessage 监听器来接收来自 AudioProcessor 的消息
            this.audioProcessor.port.onmessage = (event) => {
                if (event.data.type === 'audioprocess') {
                    // console.log("AudioRecorder: 收到 audioData chunk."); // 调试日志
                    this.audioQueue.push(event.data.audioData);
                } else if (event.data.type === 'volume_update') {
                    // console.log("AudioRecorder: 收到 volume_update."); // 调试日志
                    if (this.onVolumeUpdateCallback) {
                        this.onVolumeUpdateCallback(event.data.currentVolume, event.data.peakVolume, event.data.averageVolume);
                    }
                   
                }
            };
            console.log("AudioRecorder: AudioProcessor port.onmessage 监听器设置成功。");

            // 启动定时器发送数据
            this.sendTimer = setInterval(() => {
                if (this.audioQueue.length > 0 && this.wsClient && this.wsClient.isConnected()) {
                    const chunk = this.audioQueue.shift();
                    if (chunk) {
                        const int16Array = this._float32ToInt16(chunk);
                        this.wsClient.send(int16Array.buffer);
                        //console.log(`AudioRecorder: 发送了 ${int16Array.byteLength} 字节音频数据。`);
                    }
                }
            }, this.sendInterval);
            console.log("AudioRecorder: 音频发送定时器启动。");

            this.isRecording = true;
            console.log("AudioRecorder: 启动成功。");

        } catch (error) {
            // ！！！ 这里将打印出更具体的错误对象 ！！！
            console.error("⛔️ AudioRecorder.start() 内部错误:", error.name, error.message, error);
            this.stop(); // 确保停止所有资源
            throw error; // 重新抛出错误，以便 meeting_ui_handler 也能捕获到
        }
    }

    stop() {
        if (!this.isRecording) {
            console.warn("AudioRecorder: 录音未在进行中。");
            return;
        }

        if (this.sendTimer) {
            clearInterval(this.sendTimer);
            this.sendTimer = null;
        }

        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            // 在 AudioWorkletNode 停止时，也应停止其 port
            this.audioProcessor.port.close();
            this.audioProcessor = null;
        }
        if (this.microphone) {
            this.microphone.disconnect();
            // 停止媒体流轨道以释放麦克风
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
                this.mediaStream = null;
            }
            this.microphone = null;
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().then(() => {
                console.log("AudioRecorder: AudioContext 已关闭。");
                this.audioContext = null;
            }).catch(e => {
                console.error("AudioRecorder: 关闭 AudioContext 失败:", e);
            });
        }

        this.audioQueue = []; // 清空队列
        this.isRecording = false;
        console.log("AudioRecorder: 已停止。");
    }

    // 辅助方法：将 Float32Array 转换为 Int16Array
    _float32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            // 将 Float32 范围 -1.0 到 1.0 映射到 Int16 范围 -32768 到 32767
            int16Array[i] = Math.max(-1, Math.min(1, float32Array[i])) * 0x7FFF;
        }
        return int16Array;
    }

    // _processVolume 方法已移至 audio-processor.js 内部处理音量，并通过 port 发送更新
    // 因此这里不再需要 _processVolume 方法。
}
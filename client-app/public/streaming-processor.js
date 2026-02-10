// Resampling function (can be a shared utility, but for simplicity, duplicating it here)
const resampleAudio = (audioBuffer, originalSampleRate, targetSampleRate) => {
    if (originalSampleRate === targetSampleRate) {
        return audioBuffer;
    }
    const ratio = targetSampleRate / originalSampleRate;
    const newLength = Math.round(audioBuffer.length * ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const index = i / ratio;
        const a = Math.floor(index);
        const b = a + 1;
        const fraction = index - a;
        const valA = audioBuffer[a];
        const valB = b < audioBuffer.length ? audioBuffer[b] : valA;
        result[i] = valA + (valB - valA) * fraction;
    }
    return result;
};

class StreamingProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.originalSampleRate = 0;
        this.targetSampleRate = 16000;
        this.port.onmessage = (event) => {
            if (event.data.type === 'init') {
                this.originalSampleRate = event.data.originalSampleRate;
                this.targetSampleRate = event.data.targetSampleRate;
            }
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || input.length === 0 || this.originalSampleRate === 0) {
            return true;
        }
        const inputBuffer = input[0];
        const resampledData = resampleAudio(inputBuffer, this.originalSampleRate, this.targetSampleRate);
        const pcm16 = new Int16Array(resampledData.length);
        for (let i = 0; i < resampledData.length; i++) {
            pcm16[i] = Math.max(-1, Math.min(1, resampledData[i])) * 0x7FFF;
        }
        if (pcm16.length > 0) {
            this.port.postMessage({ type: 'audioData', pcm16Data: pcm16 }, [pcm16.buffer]);
        }
        return true;
    }
}

registerProcessor('streaming-processor', StreamingProcessor);
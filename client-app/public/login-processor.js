class LoginProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.audioBuffer = [];
        this.port.onmessage = (event) => {
            if (event.data.type === 'getAudioBuffer') {
                this.port.postMessage({ type: 'audioBuffer', audioBuffer: this.audioBuffer });
                this.audioBuffer = [];
            }
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || input.length === 0) {
            return true;
        }
        const inputBuffer = input[0];
        this.audioBuffer.push(new Float32Array(inputBuffer));
        return true;
    }
}

registerProcessor('login-processor', LoginProcessor);
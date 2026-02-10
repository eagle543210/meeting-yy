// /static/js/audio_processor_loader.js

let audioProcessorLoaded = false;

// 确保在加载模块时，audioContext 是一个有效的 AudioContext 实例
export async function loadAudioProcessor(audioContext) {
    if (audioProcessorLoaded) {
        console.log("AudioWorklet 处理器已加载。");
        return;
    }
    
    if (!audioContext || !(audioContext instanceof (window.AudioContext || window.webkitAudioContext))) {
        console.error("加载 AudioWorklet 处理器失败: audioContext 参数无效。");
        throw new Error("Invalid AudioContext provided to loadAudioProcessor.");
    }

    try {
        // audioWorklet.addModule 应该在 AudioContext 的工作线程中加载模块
        await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');
        audioProcessorLoaded = true;
        console.log("AudioWorklet 处理器加载成功！");
    } catch (error) {
        console.error("加载 AudioWorklet 处理器失败:", error);
        // 如果加载失败，需要确保捕获并向上层抛出错误，以便UI层处理
        throw new Error(`无法加载 AudioWorklet 处理器: ${error.message}`);
    }
}
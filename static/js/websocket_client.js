// /static/js/websocket_client.js

export class WebSocketClient {
    // 构造函数现在接收一个 url 参数，以及两个回调函数
    constructor(url, onMessageCallback, onStatusUpdateCallback) {
        this.url = url; // 将传入的 URL 存储为实例属性
        this.socket = null; // WebSocket 实例
        this.onMessage = onMessageCallback; // 处理接收到的消息的回调
        this.onStatusUpdate = onStatusUpdateCallback; // 更新连接状态的回调
        this.messageQueue = []; // 用于缓冲在连接断开时发送的消息
        // this.isConnected = false; // 此行可以移除，因为 isConnected() 方法会动态检查状态
        this.reconnectAttempts = 0; // 当前重连尝试次数
        this.maxReconnectAttempts = 5; // 最大重连尝试次数
        this.reconnectDelay = 1000; // 首次重连延迟（毫秒）
    }

    /**
     * 尝试建立 WebSocket 连接。
     * 如果已连接，则直接解决 Promise。
     * 支持重连机制。
     * @returns {Promise<void>} resolve 当连接建立，reject 当连接失败或达到最大重连次数。
     */
    connect() {
        return new Promise((resolve, reject) => {
            // 如果已经连接，直接返回
            if (this.isConnected()) {
                console.log("WebSocket 已经连接。");
                this.onStatusUpdate('已连接', this.messageQueue.length, 'success');
                resolve();
                return;
            }

            // 如果存在旧的 socket 实例，清理其事件监听器并尝试关闭它
            if (this.socket) {
                this.socket.onopen = null;
                this.socket.onmessage = null;
                this.socket.onclose = null;
                this.socket.onerror = null;
                if (this.socket.readyState !== WebSocket.CLOSED) {
                    this.socket.close(); // 尝试关闭旧连接
                }
                this.socket = null; // 清理旧引用
            }

            // 使用实例的 URL 创建新的 WebSocket 连接
            this.socket = new WebSocket(this.url);

            // 连接打开事件处理
            this.socket.onopen = () => {
                console.log("WebSocket 连接已建立。");
                // this.isConnected = true; // 不需要显式设置此属性，isConnected() 方法已涵盖
                this.reconnectAttempts = 0; // 重置重连次数
                this.onStatusUpdate('已连接', this.messageQueue.length, 'success');
                this._processQueue(); // 连接成功后发送队列中的消息
                resolve(); // 解决 Promise，表示连接成功
            };

            // 接收消息事件处理
            this.socket.onmessage = (event) => {
                try {
                    // 根据数据类型处理消息：二进制或文本（JSON）
                    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
                        this.onMessage({ type: 'binary_data', data: event.data });
                    } else {
                        const data = JSON.parse(event.data);
                        this.onMessage(data); // 传递解析后的 JSON 数据
                    }
                } catch (e) {
                    console.error("解析 WebSocket 消息失败:", e, event.data);
                }
            };

            // 连接关闭事件处理
            this.socket.onclose = (event) => {
                console.warn("WebSocket 连接已关闭:", event.code, event.reason);
                // this.isConnected = false; // 不需要显式设置此属性
                this.socket = null; // 在 close 事件中安全地将 socket 设为 null
                this.onStatusUpdate('断开', this.messageQueue.length, 'danger');

                // 只有在非正常关闭且未达到最大重连次数时才尝试重连
                if (event.code !== 1000 && event.code !== 1001 && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`尝试重连 (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                    this.onStatusUpdate(`重连中 (${this.reconnectAttempts})...`, this.messageQueue.length, 'warning');
                    // 递增重连延迟，并递归调用 connect，将 Promise 的 resolve/reject 向上传递
                    setTimeout(() => this.connect().then(resolve).catch(reject), this.reconnectDelay * this.reconnectAttempts);
                } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    console.error("达到最大重连次数，放弃重连。");
                    reject(new Error("WebSocket 无法连接，已达最大重连次数。"));
                } else {
                    reject(new Error("WebSocket 连接正常关闭或用户离开。")); // 正常关闭或用户离开
                }
            };

            // 错误事件处理
            this.socket.onerror = (error) => {
                console.error("WebSocket 错误:", error);
                // this.isConnected = false; // 不需要显式设置此属性
                this.onStatusUpdate('错误', this.messageQueue.length, 'danger');
                reject(error); // 发生错误，拒绝 Promise
            };
        });
    }

    /**
     * 发送数据到 WebSocket 服务器。
     * 如果未连接，则将消息放入队列。
     * @param {ArrayBuffer|Blob|Object} data 要发送的数据（二进制或JSON对象）。
     */
    send(data) {
        if (this.isConnected()) {
            if (data instanceof ArrayBuffer || data instanceof Blob) {
                this.socket.send(data); // 直接发送二进制数据
            } else {
                this.socket.send(JSON.stringify(data)); // 发送 JSON 字符串
            }
            this.onStatusUpdate('已连接', this.messageQueue.length, 'success'); // 假设发送成功则连接正常
        } else {
            console.warn("WebSocket 未连接，消息已入队列:", data);
            this.messageQueue.push(data); // 入队列，等待连接恢复后发送
            this.onStatusUpdate('断开', this.messageQueue.length, 'danger'); // 状态更新
        }
    }

    /**
     * 处理消息队列，在连接建立后发送所有排队的消息。
     * @private
     */
    _processQueue() {
        while (this.messageQueue.length > 0 && this.isConnected()) {
            const message = this.messageQueue.shift();
            this.send(message); // 从队列发送消息
        }
    }

    /**
     * 断开 WebSocket 连接。
     * @param {number} code - WebSocket 关闭代码 (可选，默认为 1000 正常关闭)。
     * @param {string} reason - 关闭原因 (可选)。
     */
    disconnect(code = 1000, reason = "客户端主动断开") {
        if (this.socket) {
            this.socket.close(code, reason);
            this.socket = null; // 清理引用
            // this.isConnected = false; // 不需要显式设置此属性
            this.messageQueue = []; // 清空队列
            console.log("WebSocket 已断开。");
        }
    }

    /**
     * 检查 WebSocket 连接是否处于打开状态。
     * @returns {boolean} 如果连接打开则为 true，否则为 false。
     */
    isConnected() {
        // 检查 socket 实例是否存在且其 readyState 是否为 OPEN
        return this.socket && this.socket.readyState === WebSocket.OPEN;
    }
}
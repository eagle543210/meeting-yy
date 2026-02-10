# Meeting Intelligent Analysis System (MIAS) | AI会议智能分析系统

![AI](https://img.shields.io/badge/AI-Intelligence-blueviolet)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![React](https://img.shields.io/badge/Frontend-React-blue)
![Database](https://img.shields.io/badge/DB-Multi--Layer-black)

MIAS 是一款基于深度学习技术构建的新一代会议智能分析系统。它能够实时处理会议音频，实现高精度转录、自动会议纪要生成、声纹识别以及基于知识图谱的闭环深度问答。

---

## 🌟 核心功能 (Core Features)

### 🎙️ 实时音频处理
*   **高精度转录 (STT)**：集成 `Faster-Whisper` 模型，提供极速且准确的语音转文字服务。
*   **智能静音检测 (VAD)**：采用 `Silero VAD` 与 `WebRTC VAD` 双重校验，精准过滤背景噪音与非人声段落。
*   **声纹识别与识别 (Diarization)**：基于 `Pyannote.audio` 技术，实现“听音识人”，精准标识不同发言者。

### 📝 会议深度分析
*   **自动纪要生成**：利用 `BART` 模型自动提炼会议重点，生成条理清晰的会议摘要。
*   **决策与任务提取**：基于 LLM 技术自动识别会议中的决策项 (Decisions) 和待办任务 (Action Items)。
*   **实时转录流**：通过 WebSocket 提供低延迟的实时文字反馈。

### 🧠 知识大脑 (RAG & KG)
*   **混合搜索架构**：结合 `Milvus` 向量数据库（语义检索）与 `Neo4j` 知识图谱（关系检索），支持复杂的历史会议查询。
*   **语义增强 (BGE)**：利用 `BGE-M3` 嵌入模型精准计算文本语义向量。
*   **LLM 交互**：内置对 `Gemini` 及本地 LLM (Llama-cpp) 的支持，提供智能问答与知识追溯。

### 👤 用户与安全
*   **声纹登录**：独特的语音注册与登录机制，无需密码，安全便捷。
*   **权限管理**：精细化的角色访问控制 (RBAC)，确保会议敏感数据安全。

---

## 🛠️ 技术栈 (Tech Stack)

| 维度 | 技术选型 |
| :--- | :--- |
| **前端** | React 19, Vite, Tailwind CSS, Lucide Icons |
| **后端** | FastAPI (Python 3.10+), Uvicorn |
| **语音模型** | Faster-Whisper, Pyannote.audio, Silero VAD |
| **NLP 模型** | BART (Summarization), BGE-M3 (Embedding), Gemini / Llama-cpp (LLM) |
| **数据库** | MongoDB (元数据), Milvus (向量), Neo4j (图谱) |
| **基础设施** | Docker Compose, WebSocket, FFmpeg |

---

## 💻 环境准备 (Environment)

### 硬件建议
*   **显存**：建议 8GB+ (用于流畅运行 Whisper Large 及 LLM)
*   **存储**：预留 20GB+ (用于模型存储及数据库索引)

### 软件依赖
*   [Python 3.10+](https://www.python.org/)
*   [Node.js 18+](https://nodejs.org/)
*   [Docker & Docker Compose](https://www.docker.com/)
*   [FFmpeg](https://ffmpeg.org/) (必须安装并添加到环境变量路径)

---

## 🚀 快速开始 (Getting Started)

### 1. 克隆项目
```bash
git clone https://github.com/eagle543210/meeting-yy.git
cd meeting-yy
```

### 2. 基础设施启动 (Docker)
确保 Docker 已启动，运行以下命令初始化数据库环境：
```bash
docker-compose up -d
```

### 3. 后端环境配置
1. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. 配置环境变量：复制 `.env.example` 为 `.env` 并填入您的 API Keys (如 Gemini API Key) 和数据库连接串。

### 4. 前端构建
```bash
cd client-app
npm install
npm run dev
```

### 5. 启动后端
```bash
python app.py
```
访问 `http://localhost:8000` 即可开始体验。

---

## 📂 项目结构 (Project Structure)

*   `core/`: 系统核心算法逻辑（STT, VAD, KG 生成等）
*   `backend/`: 后端协调器与连接管理
*   `services/`: 数据库与第三方 API 封装服务
*   `client-app/`: 基于 React 的响应式前端界面
*   `config/`: 系统全局配置项
*   `models/`: (Git 忽略) 本地 AI 模型权重存储

---

## 📧 联系与支持
如有任何问题或建议，欢迎提交 Issue 或联系支持：`459880255@qq.com`

---
*太初量化（武汉）AI可以有限公司*

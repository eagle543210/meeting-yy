// M:\meeting\client-app\src\main.jsx (或 main.js)

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx'; // 引入您的 App 组件
import './index.css'; // 引入 Vite 默认的 CSS 文件，用于 Tailwind CSS

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode> // <-- 注意：这里移除了逗号
);
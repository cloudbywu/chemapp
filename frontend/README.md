# ChemApp frontend

🌐 **语言 / Language:** [简体中文](README.md) · [English](README.en.md)

React、TypeScript 与 Vite 前端。开发时由 Vite 将 `/api` 代理到
`http://127.0.0.1:8000`。

```powershell
npm ci
npm run dev
```

浏览器访问 <http://127.0.0.1:3000>。

质量检查：

```powershell
npm run lint
npm test
npm run build
npm audit --omit=dev
```

运行时访问令牌和外部 AI 密钥只保存在内存中。不要把密钥写入源码、构建变量或
浏览器持久化存储。

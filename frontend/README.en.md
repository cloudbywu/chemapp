# ChemApp frontend

🌐 **Language / 语言:** [English](README.en.md) · [简体中文](README.md)

React, TypeScript, and Vite frontend. During development, Vite proxies `/api`
to `http://127.0.0.1:8000`.

```powershell
npm ci
npm run dev
```

Open <http://127.0.0.1:3000> in a browser.

Quality checks:

```powershell
npm run lint
npm test
npm run build
npm audit --omit=dev
```

Runtime access tokens and external AI keys are kept only in memory. Do not
write secrets into source code, build variables, or browser persistent storage.

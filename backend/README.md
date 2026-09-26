# ChemApp backend

🌐 **语言 / Language:** [简体中文](../README.md) · [English](../README.en.md)

FastAPI 后端。完整使用说明见仓库根目录 [`README.md`](../README.md) 与
[`README.en.md`](../README.en.md)（英文版）。

主要结构：

- `app/`：FastAPI 应用与业务逻辑；
- `app/ml/`：NMR 候选排序、前向评分、校准与盲测协议；
- `research/`：研究路径代码（校准训练、基线、benchmark）；
- `scripts/`：数据构建、评估与发布工具；
- `vendor/`：内置第三方代码与 CSP5 权重（见 `vendor/README.md`）；
- `archive/`：历史一次性脚本，仅用于追溯。

开发启动：

```powershell
cd backend
uv sync --locked --group dev
uv run python dev_server.py
```

质量检查：

```powershell
uv run ruff check .
uv run pytest tests -q
uv lock --check
```

部署、配置变量与安全说明见根 README。

---

FastAPI backend. For full usage instructions, see the repository root
[`README.md`](../README.md) (Chinese) or
[`README.en.md`](../README.en.md) (English).

Main layout:

- `app/`: FastAPI application and business logic;
- `app/ml/`: NMR candidate ranking, forward scoring, calibration, and blind-test protocol;
- `research/`: research-path code (calibration training, baselines, benchmarks);
- `scripts/`: data build, evaluation, and release tooling;
- `vendor/`: vendored third-party code and CSP5 weights (see `vendor/README.md`);
- `archive/`: historical one-off scripts retained for traceability only.

Development startup:

```powershell
cd backend
uv sync --locked --group dev
uv run python dev_server.py
```

Quality checks:

```powershell
uv run ruff check .
uv run pytest tests -q
uv lock --check
```

Deployment, configuration variables, and security notes are in the root README.

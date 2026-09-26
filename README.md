# ChemApp

🌐 **语言 / Language:** [简体中文](README.md) · [English](README.en.md)

ChemApp 是一个本地优先的仪器数据解析、谱图处理和证据审阅平台。项目包含
FastAPI 后端与 React/TypeScript 前端，支持中英文界面、交互式谱图、人工复核、
结果版本、跨谱图比较、报告导出和实验性的 NMR 结构候选排序。

> 结构推测结果是候选排序，不是化合物鉴定。最终结论仍需结合原始谱图、
> 积分、多重性、二维 NMR、质谱、标准品或其他独立证据确认。

## 已支持的数据

| 技术 | 输入格式 | 主要能力 |
| --- | --- | --- |
| ¹H/¹³C NMR | JEOL Delta `.jdf`、Bruker 目录或 `.zip` | 复数 FID、数字滤波补偿、FFT、相位/基线、定标、平滑、裁剪、积分和多重峰 |
| UV-Vis | `.csv`、`.txt` | 峰检测、λmax、标准曲线 |
| 荧光 | JCAMP-DX `.dx` | 激发/发射峰与 Stokes 位移 |
| XRD | `.asc`、`.ras` | d 间距与 Scherrer 晶粒尺寸 |
| HPLC | OpenLab `.dx`，或包含 `.dx` 与 `.rx`/`.acaml` 的 ZIP | 多通道解析、峰面积、保留时间、批量比较 |
| 电化学 | CHI 文本 | CV/EIS 指标 |

仓库内的 [`dataexample`](./dataexample) 可用于本地验证，其中包含 JEOL JDF 和
Bruker NMR 示例。

## 本地运行

要求：

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 22+ 与 npm

首次安装后端依赖：

```powershell
cd backend
uv sync --locked --group dev
```

启动后端：

```powershell
cd backend
uv run python dev_server.py
```

另开一个终端，安装并启动前端：

```powershell
cd frontend
npm ci
npm run dev
```

然后打开 <http://127.0.0.1:3000>。API 文档位于
<http://127.0.0.1:8000/docs>。

开发服务器默认只监听回环地址。未配置令牌时，只有本机请求可以访问 API 和
管理员操作。

## 配置与安全

复制示例配置：

```powershell
Copy-Item .env.example .env
```

重要变量：

| 变量 | 用途 |
| --- | --- |
| `CHEMAPP_DB_PATH` | SQLite 数据库位置 |
| `CHEMAPP_ACCESS_TOKEN` | 普通 API 访问令牌 |
| `CHEMAPP_ADMIN_TOKEN` | 删除、训练和模型导入等操作的独立管理员令牌 |
| `CHEMAPP_AI_PREVIEW_SECRET` | 所有后端 worker 共享的 AI 破坏性动作预览签名密钥；生产环境应显式设置 |
| `CHEMAPP_AI_PREVIEW_TTL_SECONDS` | AI 预览授权 token 有效期（秒，默认 300，最大 3600） |
| `CHEMAPP_REVIEW_ADMIN_SUBJECT` | 写入复核审计链的稳定管理员化名；不得与任何复核人主体重合 |
| `CHEMAPP_REVIEWER_TOKENS` | 服务端复核人身份到独立高熵令牌的 JSON 映射；未配置时复核写入关闭 |
| `CHEMAPP_REVIEW_ALLOWED_LICENSES` | 可进入 Gold 复核流程的 SPDX 许可证白名单 |
| `CHEMAPP_LOCAL_ACCESS_BYPASS` | 是否允许无令牌的本机访问 |
| `CHEMAPP_LOCAL_ADMIN_BYPASS` | 是否允许无管理员令牌的本机特权操作 |
| `CHEMAPP_CORS_ORIGINS` | 允许的浏览器源，逗号分隔 |
| `CHEMAPP_LLM_ALLOWED_HOSTS` | 可连接的外部 LLM 主机白名单 |
| `CHEMAPP_NMR_INDEX` | NMR 候选检索索引 |
| `CHEMAPP_NMR_INDEX_V2` | 逐谱、可追溯的 NMR v2 审计索引（compose 已只读挂载；尚不直接用于生产排序） |
| `CHEMAPP_NMR_RANKER` | NMR 排序器文件 |
| `CHEMAPP_T5_MODEL_DIR` | 实验性 SMILES 生成模型目录 |
| `CHEMAPP_DP5Q_MODE` | 可选 DP5q ¹³C 前向诊断；仅接受 `off` 或 `shadow` |
| `CHEMAPP_DP5Q_QUANTILE_MODE` | 官方 99 分位模型的管理员只读诊断；独立开关，默认 `off` |
| `CHEMAPP_DP5Q_PYTHON` / `CHEMAPP_DP5Q_REPO` | 隔离 Python 与固定 DP5 仓库路径 |

前端设置中的 ChemApp 访问令牌、管理员令牌、谱图复核人令牌和外部 AI API Key 仅保存在当前
页面内存中；刷新页面后需要重新输入。外部 AI 功能会先要求用户确认发送数据。

远程部署时必须：

1. 为访问令牌和管理员令牌设置不同的强随机值。
2. 将两个本机绕过变量设为 `0`。
3. 使用 HTTPS 和受信任的反向代理。
4. 把数据库、原始谱图、模型和密钥放在源码目录之外并单独备份。
5. 确认 `CHEMAPP_NMR_INDEX_V2` 指向实际挂载的 NMR v2 审计索引（compose 已预置只读挂载）。

多 worker 部署必须为 `CHEMAPP_AI_PREVIEW_SECRET` 配置相同的稳定强随机值。
未配置时服务端会依次从访问/管理员令牌派生；轮换这些密钥会立即使尚未执行的
预览 token 失效。仅单进程本地开发和测试会安全地使用进程随机回退。

谱级 Gold 人工复核采用两名不同的服务端认证主体、不可变谱图/结果哈希、
冲突裁决与逐事件审计。该清单只可作为校准候选池，不能绕过分子/骨架/来源
泄漏检查而充当独立测试集。

更多说明见 [`SECURITY.md`](./SECURITY.md) 和
[`THIRD_PARTY_DATA.md`](./THIRD_PARTY_DATA.md)。

## ML 模块与 NMR 结构候选排序

本仓库内置一套用于 NMR 结构候选排序的 ML 模块：

- `backend/app/ml/forward_v1/`：可运行的 ¹³C 原子级 GNN 前向原型
  （纯 PyTorch，无 torch_geometric 依赖）与
  `csp5_scorer.py`（vendored CSP5q-13C 评分器，`csp5` 包见
  `backend/vendor/csp5/`，MIT）；
- **生产路径**：`backend/app/ml/forward_v1/`（CSP5q-13C 与本地 GNN 前向）、
  `nmr_hybrid_predictor.py`（混合排序 + 受策略门禁的条件概率执行路径）、
  `nmr_candidate_generation_v1/v2.py`、`nmr_evidence.py`、
  `nmr_structure_elucidation.py`、`app/ml/calibration/`（冻结校准器与策略）、
  `nmr_blind_challenge.py`（盲测协议工具链）；
- **研究路径**：`backend/research/`（校准训练/案例、基线、benchmark、
  独立数据）与 `app/ml` 中冻结的 v4/v5 研究 release
  （`nmr_calibration_v4/v5.py`、`nmr_v4_experiment.py`、
  `nmr_phase6_independent_test.py` 等）。

说明：条件概率校准与自动结构选择的生产门禁当前保持关闭（失败关闭）；相关
研究治理工件（评估报告、模型卡、发布清单、泄漏审计、盲测轮次记录与外部持有
交接包）不在本公开仓库中。

生产前向模型实测（2026-08-06）：

- Exp22K 官方 scaffold-DOI test（5,188 分子）：¹³C MAE **0.59 ppm**、
  RMSE 1.05、q10–q90 覆盖率 91.5%；
- NMRexp 人工复核严格 ¹³C 子集（132 条，与本地 nmrshiftdb2 索引无重叠）：
  平均谱级 MAE **0.75 ppm**，93.1% 记录 ≤ 2.0 ppm；但 CSP5 split 审计将
  130 条映射回官方 parquet，其中 49 条位于 scaffold-DOI `train`，所以该结果
  **不独立于 CSP5 前向模型训练**，仅作为回顾性同域诊断。

部署决策（2026-08-07，已确认并落地）：

- **CSP5 默认启用**：Dockerfile/compose 设 `CHEMAPP_CSP5_MODE=on`；本地默认
  `auto`（有权重则用，否则警告并回退）；
- **权重打进镜像**：`COPY backend/vendor ./vendor` 包含 CSP5 权重
  （~73 MB），`.dockerignore` 不再排除 `vendor/**/*.pt`；干净检出时
  Docker build 阶段自动从固定 PyPI sdist 下载并按 SHA-256 校验
  （`scripts/fetch_csp5_weights.py`），本地也可手动运行该脚本恢复；
- **GPU 优先、CPU 可退**：scorer 自动选择 CUDA/CPU，无需额外开关；
- **单 worker**：uvicorn `--workers 1`，评分器按进程缓存；
- **启动哈希校验失败关闭**：`backend/vendor/csp5/weights-manifest.json`
  固化 4 个权重的 SHA-256；`CHEMAPP_CSP5_MODE=on` 时校验失败应用拒绝启动，
  `auto` 时仅告警回退（见 `app/ml/deployment_check.py`）。

候选生成与端到端（2026-08-06）：

- 混合生成（本地索引 + PubChem 分子式）在 450 条 NMR-Solver 回顾集上把
  精确覆盖率从 1.8% 提升到 **38.0%**（连接性 46.4%）；
- 端到端排序（v2 计数感知预排序）：总体 Top-1 23.5%、MRR 0.246；真值在池内
  时 Top-1 **87.1%**；校准门禁 ECE 0.153 超标，保持失败关闭。

盲测与封闭池评估：

- 仓库内置完整盲测工具链：`nmr_blind_challenge.py` 支持角色分离、Gold 承诺、
  Ed25519 签名释放与逐事件审计；与之配合的谱级 Gold 人工复核流程见上文
  “配置与安全”。
- 已对历史数据完成多轮回顾性封闭池评估（v1–v7，每轮 100–500 条 ¹³C 谱）：
  真值位于候选池内时 Top-1 约 **85–87%**（含池外真值的总体口径则明显更低）；
  但各轮选样与 CSP5 训练数据存在重叠，且仓库没有真实外部持有方数据，因此这些
  结果均**不构成独立测试准确率**，只说明同一回顾性协议下的工程稳定性。
- 校准概率执行路径已经接线，但生产 gate 保持关闭：hybrid predictor/API/前端
  具备条件 Top-1 概率字段，`probability_claim_allowed=false` 时不得把它解释或
  展示为生产校准置信度；自动结构选择同样关闭，直至新的、训练隔离且真正外部
  持有的 holdout 完成。

原型冒烟训练（CPU，约 1 分钟）：

```powershell
cd backend
.venv\Scripts\python.exe -m app.ml.forward_v1.train `
  --max-molecules 5000 --epochs 60 --outdir reports/forward_v1_smoke
```

## Docker Compose

先复制 `.env.example` 为 `.env`，替换其中的访问令牌与管理员令牌（如启用
谱图复核，再替换复核人令牌），再执行：

```powershell
docker compose up --build
```

访问 <http://127.0.0.1:3000>。Compose 只将前端暴露到本机，后端由前端容器
反向代理。首次启动前确认下列模型文件存在，或者按需删除相应只读挂载：

- `backend/data/nmr_spectral_index.sqlite`
- `backend/data/nmr_spectral_index_v2.sqlite`
- `backend/data/nmr_joint_ranker.joblib`
- `backend/app/ml/pretrained/t5_nmr`


### 容器本地测试（wslc）

本仓库的容器测试统一使用 WSL 内置的 `wslc`（需 WSL 2.9.3+；本机二进制位于
`C:\Program Files\WSL\wslc.exe`），不使用 Docker Desktop。首次执行可先运行
`wslc settings reset` 初始化设置。

构建与启动后端：

```powershell
wslc build -t chemapp-backend:test -f backend/Dockerfile .
wslc run --rm -d --name chemapp-backend-test -p 127.0.0.1:8001:8000 `
  -e CHEMAPP_ACCESS_TOKEN=test-access -e CHEMAPP_ADMIN_TOKEN=test-admin `
  -e 'CHEMAPP_REVIEWER_TOKENS={"reviewer1":"test-reviewer-token"}' -e CHEMAPP_CSP5_MODE=on `
  -v "$PWD\backend\data\nmr_spectral_index.sqlite:/var/lib/chemapp-models/nmr_spectral_index.sqlite:ro" `
  -v "$PWD\backend\data\nmr_spectral_index_v2.sqlite:/var/lib/chemapp-models/nmr_spectral_index_v2.sqlite:ro" `
  -v "$PWD\backend\data\nmr_joint_ranker.joblib:/var/lib/chemapp-models/nmr_joint_ranker.joblib:ro" `
  -v "$PWD\backend\app\ml\pretrained\t5_nmr:/var/lib/chemapp-models/t5_nmr:ro" `
  chemapp-backend:test
```

验证：

```powershell
curl.exe http://127.0.0.1:8001/api/ready
curl.exe -H "X-ChemApp-Access-Token: test-access" http://127.0.0.1:8001/api/health
```

- 无令牌访问 `/api/health` 应返回 401；带令牌时应看到 `csp5_weights.status == "ok"`
  与 `nmr_index_v2.exists == true`。
- docker.io 直连可能被重置，可先用 `wslc pull docker.m.daocloud.io/library/python:3.11-slim`
  并 `wslc tag docker.m.daocloud.io/library/python:3.11-slim python:3.11-slim` 后本地构建。
- 若走代理，需在 `%USERPROFILE%\.wslconfig` 启用 `networkingMode=mirrored` 与
  `autoProxy=true` 后 `wsl --shutdown` 重启。

## NMR 处理原则

- 导入时保留不可变的处理源、来源哈希和复数正交数据（若仪器文件提供）。
- “预览”不会写数据库；“应用”需要匹配 `spectrum_revision`，并产生新修订号。
- “重置”从不可变源重放，避免多次处理造成累积失真。
- 相位校正要求复数正交数据。历史上仅保存实数的数据会明确拒绝相位操作，
  而不是伪造校正结果。
- 峰检测、积分与展示保留正负信号；指标使用带符号数据，不再把负峰静默截断。
- ¹H 自动合组会先逐峰移除已标注的溶剂线，并保留 S/N ≥ 8 的孤立 singlet；
  自动标签带来源和 S/N 审计字段，不冒充人工峰归属。

## 结构候选推测

当前流程：

1. 严格解析并规范化分子式，计算 DBE。
2. 清理溶剂/TMS 信号，合并过近谱线，并保留积分、多重性和归属信息。
3. 先按分子式筛选数据库候选，再使用 ¹H/¹³C 一对一峰匹配进行排序。
4. 只有采用 Bemis–Murcko 骨架分组验证训练的排序器才会启用；不兼容的旧模型
   会自动禁用。
5. T5 生成结果单列为“实验性假设”，必须通过 RDKit 结构与分子式校验，且不为
   数据库候选加分。

响应会区分 `ranking_score`、证据等级、预处理记录和警告，不把排序分数冒充
校准概率。没有足够独立证据时，混合物分析会选择弃权。

可追溯的 v2 审计索引以“单张谱图”为单位保存实验条件、来源快照、许可证和
导入判定，不会把同一分子的不同实验条件合并成一张伪谱；来源哈希与快照
绑定在写库前校验，不匹配会失败。缺少明确计算方法标签的记录会被单独标为
`inferred_measured`，这类记录只能进入探索性评估，不能与明确标注的
`measured` 测试集混合报告。

benchmark 评估要求模型、校准器、运行配置和去库证明先预登记并绑定哈希，再
交给评估流程（工具位于 `backend/research/`）。未经明确人工复核的记录只能
作为协议冒烟和探索基线，不能作为生产准确率依据。

### 可选的 DP5q ¹³C 影子评估

DP5q 与主后端的 NumPy/TensorFlow 依赖不兼容，因此必须使用隔离环境运行：
`CHEMAPP_DP5Q_PYTHON` 与 `CHEMAPP_DP5Q_REPO` 分别指向隔离 Python 与固定的
DP5 仓库路径，上游 commit、模型、preprocessor 与依赖版本均已固定。该隔离环境
不是 OS 级沙箱，生产部署仍应外加无网络、只读文件系统和降权策略。均值路径仍
只在 `CHEMAPP_DP5Q_MODE=shadow` 时执行；默认 `off`，而且其协议和排序行为
没有因量化模型接入而改变。

99 分位模型有独立开关 `CHEMAPP_DP5Q_QUANTILE_MODE=shadow`，仅通过管理员端点
`POST /api/ml/elucidate/dp5q/quantile-shadow` 对显式候选运行，默认最多 3 个。
当前实验峰未提供可靠原子归属，所以接口以 q50 做 Hungarian 匹配；这不是官方
assignment 语义，结果只标记为未校准诊断分数，不进入正式排序。接口只返回压缩
摘要，不返回完整 99 分位张量。

研究评估沿革：NMR 结构候选排序的研究评估经历了多个阶段——v3 NMRexp 预登记
运行、v4 NMR-Solver ¹³C 高斯集合相似度评分器、v5 多证据排序器、v8 数据隔离
审计与嵌套分组评估、v9 适用性签名与研究用校准方法比较框架。其中 v8 排序器在
仅 development 的嵌套分组评估中未优于固定参考，被标记为
`rejected_development_noninferiority`，默认拒绝加载且生产模型未变；校准域
始终失败关闭（calibration 记录为 0）。这些阶段的完整报告、发布清单、泄漏
审计与盲测轮次记录属于研究治理工件，不在本公开仓库中。数据治理与隔离检查由
`app/ml` 中的模块在导入时执行；nmrXiv 接入采用元数据优先、逐 study 有效许可
判定，未经许可、谱文件哈希和双人复核的数据不会进入校准。

## 数据完整性

- SQLite 使用 WAL、外键和忙等待；结果版本写入与当前结果更新处于同一事务。
- 谱图和结果分别带修订号。过期写入返回 HTTP 409，避免多标签页静默覆盖。
- 已人工确认的结果不会被普通重新分析覆盖；界面会要求明确确认。
- ZIP、DOCX 与 HPLC 包实施文件数、解压总量、压缩比、路径和 XML 安全检查。
- 导出的 CSV 会转义可触发电子表格公式执行的单元格。

请在升级前备份运行数据库。仓库中的 `backend/data/chemapp.db` 属于运行数据，
不应当作为可回滚的源码文件处理。

## 测试

```powershell
cd backend
uv run ruff check .
uv run pytest tests -q
uv lock --check

cd ..\frontend
npm run lint
npm test
npm run build
npm audit --omit=dev
```

持续集成配置位于 [`.github/workflows/ci.yml`](./.github/workflows/ci.yml)。

## 主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/live`、`/api/ready`、`/api/health` | 存活、就绪与详细健康状态 |
| `POST` | `/api/upload` | 上传仪器数据 |
| `GET` | `/api/spectra/{id}` | 读取谱图；不会隐式运行分析 |
| `POST` | `/api/analyze/{id}` | 显式运行分析 |
| `GET` | `/api/results/{id}` | 读取已有分析；不会隐式生成结果 |
| `PUT` | `/api/results/{id}/manual` | 保存人工复核结果 |
| `POST` | `/api/nmr/{id}/process` | 预览或应用 NMR 处理 |
| `POST` | `/api/nmr/{id}/reset` | 从原始处理源重置 |
| `POST` | `/api/ml/elucidate/predict` | NMR 候选排序 |
| `POST` | `/api/ml/elucidate/predict/combined` | 使用已保存的 ¹H/¹³C 结果排序 |
| `POST` | `/api/ml/elucidate/dp5q/quantile-shadow` | 管理员只读的 99 分位候选诊断，不参与排序 |
| `POST` | `/api/inference` | 跨技术只读推断 |
| `POST` | `/api/reports/{format}` | 导出报告 |

API 的输入模型默认拒绝未知字段，并对 ID 数量、标题、数值范围和修订号进行校验。

## 许可证

本项目以 MIT 许可证发布，见根目录 [`LICENSE`](./LICENSE)。第三方组件与数据
的署名要求见 [`NOTICE`](./NOTICE) 与 [`THIRD_PARTY_DATA.md`](./THIRD_PARTY_DATA.md)。

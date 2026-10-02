# ORPHAN CHECKPOINT — 不可加载，仅归档

`nmr_ensemble_best.pt` 是一个**孤儿 checkpoint**：

- 模型头为 **8 类**，且文件中**不含 `class_names` 元数据**；
- 当前 app 按 `nmr_ensemble.pt`（10 类功能团，自带 class_names）或数据库类序构建 10 类模型，
  strict 加载本文件必然失败；
- `predictor._ensure_model()` 仅在 `nmr_ensemble.pt` 缺失时才会回退尝试本文件（:54-56），
  届时会因类数/元数据不匹配报错——**实际不可达**。

2026-09-27 ML 评估结论：该 legacy ¹H 功能团分类链路为演进不一致的 demo 原型
（当前数据生成器分布与 checkpoint 训练分布已漂移；真实感端到端 smoke 1/6 命中），
仅作归档保留，**不要作为生产 ML 证据引用，也不要重训覆盖本文件**。评估详情见
`backend/reports/ml_eval/ML_EVALUATION_REPORT.md` §6。

如需清理磁盘（3.2MB），可安全删除本文件；删除前请确认 `nmr_ensemble.pt` 仍在位。
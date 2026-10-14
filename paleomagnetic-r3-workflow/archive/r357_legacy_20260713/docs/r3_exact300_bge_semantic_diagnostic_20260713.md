# R3 exact300 BGE 语义分支诊断（2026-07-13）

## 结论

BGE 目标条件语义缓存已经完成无标签冻结，但本次 1--3 折门控试跑作废，未产生可引用的 accuracy、report 或 OOF，不得将候选冻结回执解释为评测成绩，本分支不晋升到主流程。

## 已冻结产物

- 缓存：`tmp/r3_exact_300_resultno_v2/bge_semantic_cache_v1/features.npz`
- 缓存 SHA-256：`38e8826dbd7dab58ca21a1b0efe5eedbcf09c2b9b12c37b7576446d6f8cfd126`
- 清单 SHA-256：`bafe40529d05a836796dd7bd9817412cf4bf82a3062161e3c2cb52d2e12dd686`
- 范围：2,172 行；四个 768 维归一化视图（target query、abstract、target local、target paper）和四维 relation。
- 边界：构建器不读取标签、目标工作簿、PMAGRESULT/ROCKMAG、旧预测、旧评测或记录标识符。

## 作废原因

canonical 环境为 sklearn 1.6.1、NumPy 2.0.2 和 macOS Accelerate。第一次 randomized PCA 试跑、以及随后用于复核的 full-SVD 测试投影，都在有界且已验证 finite 的 BGE 输入上持续产生 `divide by zero`、`overflow` 和 `invalid value in matmul` 运行时告警。为避免把平台线性代数异常产生的结果登记为模型成绩，试跑被主动终止；输出目录中只有 `candidate_freeze_folds_1_2_3.json`，没有 `folds_1_2_3_report.json` 或 OOF。

当前 90% 目标已由其他 canonical 分支达到，因此不再为 BGE 执行昂贵的全 PCA/五折计算。若以后重新启用，应先在固定 OpenBLAS/Linux 运行时或经过验证的稳定投影实现上重新冻结运行协议，再从零执行 canonical 门控。

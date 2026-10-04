# SafeNet
# SafeNet AI - Frozen Research Package v1

Freeze date: **2026-10-02**

## Research scope

This package evaluates a controlled hybrid architecture for Vietnamese text scam detection on a CPU-only edge setting. The lightweight TF-IDF + Logistic Regression detector and symbolic rules carry the primary deterministic path. Semantic retrieval provides reference context, and local Qwen2.5:1.5B is invoked selectively or as an explicit ablation baseline. The SLM is not granted unrestricted decision authority in the selective safety mode.

ECL, LAM, multimodal/deepfake processing, and autonomous reporting are intentionally excluded from the core paper evaluation.

## What is already frozen

- Development corpus preprocessing and exact deduplication.
- Scenario-aware train/validation/diagnostic split.
- Independent post-freeze holdout (`160` rows; 80 benign/80 scam; zero exact overlap with development).
- OOD/gray set (`90` rows; 30 unseen scam, 30 hard-negative benign, 30 gray-area).
- Retrieval knowledge base and DEV/TEST retrieval queries.
- LR architecture and random seed.
- Gate threshold selection procedure (validation only).
- Symbolic rules v1.
- Qwen model/version and structured-output prompts in the research server.
- Ablation configurations B0-B2 and A1-A4.
- Resource and statistical analysis scripts.

## Important validity boundary

The 500-row original corpus was inspected and used during development. It is **not** a final independent test set. The independent holdout and OOD/gray sets are author-curated synthetic sets created after the demo freeze. They support controlled comparisons, not claims of population-level deployment accuracy.

## One-time installation

```powershell
cd D:\RAGECL\safenet_final_research_package
python -m pip install -r .\requirements_research.txt
ollama pull qwen2.5:1.5b
```

Keep Ollama running. On Windows desktop installations it normally runs as a background service; otherwise start:

```powershell
ollama serve
```

## Run the entire final experiment

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_final_pipeline.ps1
```

This performs, in order:

1. data preparation and scenario-aware splitting;
2. LR training and gate calibration using development data only;
3. retrieval benchmark and semantic embedding selection using retrieval DEV only;
4. frozen Chroma index construction;
5. research API startup;
6. B0/B1/B2/A1/A2/A3/A4 independent-holdout ablation;
7. A4 selective OOD/gray evaluation;
8. CPU/RAM/latency benchmark;
9. Wilson confidence intervals, paired exact McNemar tests, paired bootstrap CIs, and Holm adjustment;
10. aggregation into `results/FINAL_RESULTS_BUNDLE.json`.

## Ablation definitions

| ID | Configuration |
|---|---|
| B0 | TF-IDF + Logistic Regression |
| B1 | Qwen2.5:1.5B neutral zero-shot |
| B2 | Qwen2.5:1.5B balanced few-shot |
| A1 | LR + frozen symbolic rules |
| A2 | Semantic RAG + Qwen on every sample |
| A3 | LR + rules + selective Qwen |
| A4 | LR + rules + semantic RAG + selective Qwen (full SafeNet) |

Independent binary evaluation uses the same frozen 160-message test set for every configuration. A4 is additionally evaluated in selective mode on the 90-message OOD/gray set.

## Primary outputs

- `results/ablation_summary.csv`
- `results/ablation_predictions.csv`
- `results/ood_gray_summary.json`
- `results/rag_benchmark_results.csv`
- `results/resource_summary.csv`
- `results/statistical_comparison.csv`
- `results/accuracy_wilson_ci.csv`
- `results/FINAL_RESULTS_BUNDLE.json`
- `paper/RESULTS_TABLES.md`

## What not to report as a final result

- The old 500-row A4 experiment.
- The earlier 20-row parser/prompt smoke tests.
- The 12-case hybrid smoke result.
- The random deduplicated LR split that produced 100% accuracy.

Those runs are engineering diagnostics only.

## Paper title (working)

**Design and Evaluation of a Controlled Hybrid Architecture for Vietnamese Scam Detection on CPU-Only Edge Devices**

# SafeNet Research Dataset Card v1

## Freeze date
2026-10-02

## Development corpus
The original `dataset_vietscam_500_original.csv` contains 500 synthetic Vietnamese messages produced from fixed text templates. After exact-text deduplication, 297 usable unique messages remain. 203 exact duplicate rows are removed. The surviving rows are mapped back to their originating template/scenario and split with `StratifiedGroupKFold`, so a template/scenario cannot appear in more than one of train, validation, or scenario-diagnostic test.

This development corpus has already been inspected during system development. Therefore it is **not** treated as an independent final test set.

## Independent holdout v1
`independent_test_v1.csv` contains 160 newly authored synthetic messages (80 benign and 80 scam) across 16 scenarios. They were created only after Demo Freeze v1 and are not used for prompt, threshold, rule, or retrieval tuning. Exact-text overlap with the development corpus is zero.

The set is synthetic/author-curated rather than a naturally sampled population. Results must therefore be described as performance on a controlled independent holdout, not as population-level accuracy.

## OOD/gray-area v1
`ood_gray_v1.csv` contains 90 samples across 18 scenarios: unseen scam patterns, hard-negative benign messages, and deliberately under-specified gray-area messages. Gray-area rows use `expected_action=UNCERTAIN`; they are for selective-safety evaluation and are excluded from ordinary binary accuracy.

## Retrieval benchmark
`rag_knowledge_v1.csv` contains 12 versioned scenario records. Retrieval model/threshold selection uses only `rag_queries_dev.csv`; `rag_queries_test.csv` remains untouched until the retrieval configuration is frozen. Test metrics are Recall@1, Recall@3, MRR, and false-retrieval rate on NO_MATCH queries.

## Intended claims
The datasets support controlled comparative experiments and system ablation. They do not support claims of nationwide prevalence, deployment-grade detection accuracy, or demographic representativeness. A future journal extension should include a larger naturally sampled and independently annotated Vietnamese corpus.

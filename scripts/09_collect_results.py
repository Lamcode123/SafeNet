from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'; PAPER=ROOT/'paper'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    PAPER.mkdir(exist_ok=True)
    bundle={
        'lr_training':read_json(RESULTS/'lr_training_calibration_report.json'),
        'rag_selection':read_json(ROOT/'freeze'/'rag_selection_v1.json'),
        'ood_gray':read_json(RESULTS/'ood_gray_summary.json'),
        'environment':read_json(RESULTS/'run_environment.json'),
        'resource_environment':read_json(RESULTS/'resource_environment.json'),
    }
    if (RESULTS/'ablation_summary.csv').exists():
        bundle['ablation']=pd.read_csv(RESULTS/'ablation_summary.csv').to_dict(orient='records')
    if (RESULTS/'rag_benchmark_results.csv').exists():
        bundle['rag_benchmark']=pd.read_csv(RESULTS/'rag_benchmark_results.csv').to_dict(orient='records')
    if (RESULTS/'resource_summary.csv').exists():
        bundle['resource']=pd.read_csv(RESULTS/'resource_summary.csv').to_dict(orient='records')
    if (RESULTS/'statistical_comparison.csv').exists():
        bundle['statistics']=pd.read_csv(RESULTS/'statistical_comparison.csv').to_dict(orient='records')

    (RESULTS/'FINAL_RESULTS_BUNDLE.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')

    md=['# SafeNet Final Results Tables','']
    if bundle.get('ablation'):
        df=pd.DataFrame(bundle['ablation'])
        cols=[c for c in ['config','n','valid_rate','accuracy','precision','recall','f1','specificity','fpr','fnr','balanced_accuracy','mcc','qwen_invocation_rate','latency_median_ms','latency_p95_ms'] if c in df.columns]
        md+=['## Main binary ablation','',df[cols].to_markdown(index=False,floatfmt='.4f'),'']
    else:
        md+=['## Main binary ablation','','**PENDING:** run `06_run_ablation.py`.','']

    if bundle.get('rag_benchmark'):
        df=pd.DataFrame(bundle['rag_benchmark'])
        cols=[c for c in ['method','dev_recall_at_1','dev_recall_at_3','dev_mrr','similarity_threshold','test_recall_at_1','test_recall_at_3','test_mrr','test_false_retrieval_rate'] if c in df.columns]
        md+=['## Retrieval benchmark','',df[cols].to_markdown(index=False,floatfmt='.4f'),'']

    if bundle.get('ood_gray'):
        md+=['## OOD / gray-area safety','', '```json',json.dumps(bundle['ood_gray'],ensure_ascii=False,indent=2),'```','']
    else:
        md+=['## OOD / gray-area safety','','**PENDING:** run A4 selective benchmark.','']

    if bundle.get('resource'):
        df=pd.DataFrame(bundle['resource'])
        md+=['## CPU/RAM/latency','',df.to_markdown(index=False,floatfmt='.3f'),'']
    else:
        md+=['## CPU/RAM/latency','','**PENDING:** run `07_resource_benchmark.py`.','']

    if bundle.get('statistics'):
        df=pd.DataFrame(bundle['statistics'])
        md+=['## Paired statistical comparisons','',df.to_markdown(index=False,floatfmt='.4f'),'']
    else:
        md+=['## Paired statistical comparisons','','**PENDING:** run `08_statistics.py`.','']

    (PAPER/'RESULTS_TABLES.md').write_text('\n'.join(md),encoding='utf-8')
    print('Wrote',RESULTS/'FINAL_RESULTS_BUNDLE.json')
    print('Wrote',PAPER/'RESULTS_TABLES.md')

if __name__=='__main__': main()

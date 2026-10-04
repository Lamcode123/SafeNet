from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FREEZE = ROOT / "freeze"

SEMANTIC_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def rank_metrics(query_df, kb_ids, sim):
    match_rows = query_df[query_df.expected_scenario != "NO_MATCH"].index.tolist()
    rr=[]; r1=[]; r3=[]
    for qi in match_rows:
        expected=query_df.loc[qi,"expected_scenario"]
        order=np.argsort(-sim[qi])
        ranked=[kb_ids[j] for j in order]
        pos=ranked.index(expected)+1 if expected in ranked else 10**9
        rr.append(1.0/pos if pos < 10**9 else 0.0)
        r1.append(1.0 if pos <= 1 else 0.0)
        r3.append(1.0 if pos <= 3 else 0.0)
    return {
        "recall_at_1":float(np.mean(r1)) if r1 else None,
        "recall_at_3":float(np.mean(r3)) if r3 else None,
        "mrr":float(np.mean(rr)) if rr else None,
    }


def choose_threshold(query_df, kb_ids, sim):
    top_idx=np.argmax(sim,axis=1)
    top_sim=sim[np.arange(len(query_df)),top_idx]
    top_sid=np.array([kb_ids[i] for i in top_idx])
    expected=query_df.expected_scenario.values
    match=expected != "NO_MATCH"
    no_match=~match

    # Candidate threshold learned on retrieval-dev only.
    candidates=np.linspace(0.10,0.95,171)
    rows=[]
    for t in candidates:
        relevant_accept=((top_sid==expected) & (top_sim>=t) & match).sum()/max(1,match.sum())
        no_match_reject=((top_sim<t) & no_match).sum()/max(1,no_match.sum())
        false_retrieval=((top_sim>=t) & no_match).sum()/max(1,no_match.sum())
        balanced=(relevant_accept+no_match_reject)/2
        rows.append((balanced,relevant_accept,no_match_reject,false_retrieval,float(t)))
    rows.sort(reverse=True)
    b,ra,nr,fr,t=rows[0]
    return {
        "similarity_threshold":t,
        "dev_balanced_retrieval_score":b,
        "dev_relevant_accept_rate":ra,
        "dev_no_match_reject_rate":nr,
        "dev_false_retrieval_rate":fr,
    }


def threshold_metrics(query_df, kb_ids, sim, threshold):
    top_idx=np.argmax(sim,axis=1)
    top_sim=sim[np.arange(len(query_df)),top_idx]
    top_sid=np.array([kb_ids[i] for i in top_idx])
    expected=query_df.expected_scenario.values
    match=expected != "NO_MATCH"
    no_match=~match
    return {
        "relevant_accept_rate":float(((top_sid==expected)&(top_sim>=threshold)&match).sum()/max(1,match.sum())),
        "no_match_reject_rate":float(((top_sim<threshold)&no_match).sum()/max(1,no_match.sum())),
        "false_retrieval_rate":float(((top_sim>=threshold)&no_match).sum()/max(1,no_match.sum())),
        "mean_top1_similarity_match":float(top_sim[match].mean()) if match.any() else None,
        "mean_top1_similarity_no_match":float(top_sim[no_match].mean()) if no_match.any() else None,
    }


def evaluate(name, encoder, kb, dev, test):
    docs=(kb.title.fillna("")+". "+kb.content.fillna("")).tolist()
    dev_q=dev["query"].tolist(); test_q=test["query"].tolist()
    kb_ids=kb.scenario_id.tolist()

    if name=="TFIDF_WORD_CHAR":
        vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),sublinear_tf=True)
        vec.fit(docs+dev_q)  # dev is allowed for retrieval model selection; test remains untouched.
        D=vec.transform(docs)
        dev_sim=cosine_similarity(vec.transform(dev_q),D)
        test_sim=cosine_similarity(vec.transform(test_q),D)
    else:
        model=encoder
        D=model.encode(docs,normalize_embeddings=True,show_progress_bar=False)
        Qd=model.encode(dev_q,normalize_embeddings=True,show_progress_bar=False)
        Qt=model.encode(test_q,normalize_embeddings=True,show_progress_bar=False)
        dev_sim=np.asarray(Qd) @ np.asarray(D).T
        test_sim=np.asarray(Qt) @ np.asarray(D).T

    dev_rank=rank_metrics(dev,kb_ids,dev_sim)
    threshold=choose_threshold(dev,kb_ids,dev_sim)
    test_rank=rank_metrics(test,kb_ids,test_sim)
    test_thr=threshold_metrics(test,kb_ids,test_sim,threshold["similarity_threshold"])

    return {
        "method":name,
        **{f"dev_{k}":v for k,v in dev_rank.items()},
        **threshold,
        **{f"test_{k}":v for k,v in test_rank.items()},
        **{f"test_{k}":v for k,v in test_thr.items()},
    }


def main():
    RESULTS.mkdir(exist_ok=True); FREEZE.mkdir(exist_ok=True)
    kb=pd.read_csv(DATA/'rag_knowledge_v1.csv')
    dev=pd.read_csv(DATA/'rag_queries_dev.csv')
    test=pd.read_csv(DATA/'rag_queries_test.csv')

    rows=[]
    rows.append(evaluate('TFIDF_WORD_CHAR',None,kb,dev,test))

    semantic_errors={}
    try:
        from sentence_transformers import SentenceTransformer
        for model_name in SEMANTIC_CANDIDATES:
            try:
                print('Loading',model_name)
                model=SentenceTransformer(model_name)
                rows.append(evaluate(model_name,model,kb,dev,test))
            except Exception as e:
                semantic_errors[model_name]=repr(e)
                print('FAILED',model_name,repr(e))
    except Exception as e:
        semantic_errors['sentence_transformers_import']=repr(e)

    out=pd.DataFrame(rows)
    out.to_csv(RESULTS/'rag_benchmark_results.csv',index=False,encoding='utf-8-sig')

    semantic=out[out.method.str.startswith('sentence-transformers/')].copy()
    if len(semantic):
        # Selection uses DEV metrics only. Test columns are never used for selection.
        semantic=semantic.sort_values(
            ['dev_balanced_retrieval_score','dev_mrr','dev_recall_at_1'],
            ascending=False,
        )
        best=semantic.iloc[0].to_dict()
        selection={
            'version':'rag_selection_v1',
            'frozen_on':'2026-10-02',
            'selected_embedding_model':best['method'],
            'similarity_threshold':float(best['similarity_threshold']),
            'distance_threshold_for_chroma_cosine':float(1.0-best['similarity_threshold']),
            'selection_basis':'retrieval DEV only: balanced relevant-accept/no-match-reject, then MRR and Recall@1',
            'test_metrics_reported_after_freeze':{
                k:(None if pd.isna(v) else float(v))
                for k,v in best.items() if str(k).startswith('test_')
            },
            'semantic_errors':semantic_errors,
        }
        (FREEZE/'rag_selection_v1.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(selection,ensure_ascii=False,indent=2))
    else:
        print('No semantic model completed. Install sentence-transformers / download models, then rerun.')
        (RESULTS/'rag_semantic_errors.json').write_text(json.dumps(semantic_errors,indent=2),encoding='utf-8')

    print(out.to_string(index=False))


if __name__=='__main__':
    main()

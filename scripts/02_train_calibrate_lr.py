from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "freeze"
SEED = 20261002


def build_model():
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1,2), min_df=1,
                max_features=20000, sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3,5), min_df=1,
                max_features=30000, sublinear_tf=True,
            )),
        ])),
        ("clf", LogisticRegression(
            C=2.0, max_iter=3000, class_weight="balanced",
            random_state=SEED,
        )),
    ])


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn/(tn+fp)) if tn+fp else None,
        "fpr": float(fp/(fp+tn)) if fp+tn else None,
        "fnr": float(fn/(fn+tp)) if fn+tp else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)},
    }


def choose_gate(y, p, target_selective_accuracy=0.95):
    candidates=[]
    lows=np.arange(0.10,0.50,0.025)
    highs=np.arange(0.525,0.925,0.025)
    for low in lows:
        for high in highs:
            if low >= 0.5 or high <= 0.5 or low >= high:
                continue
            decided=(p <= low) | (p >= high)
            n=int(decided.sum())
            if n == 0:
                continue
            pred=(p[decided] >= 0.5).astype(int)
            yy=y[decided]
            sel_acc=float((pred == yy).mean())
            coverage=float(decided.mean())
            # Require at least one decided sample of each class when possible.
            has_both=len(set(yy.tolist())) == 2
            candidates.append({
                "low":float(round(low,3)), "high":float(round(high,3)),
                "selective_accuracy":sel_acc, "coverage":coverage,
                "decided_n":n, "has_both_classes":has_both,
            })
    feasible=[c for c in candidates if c["selective_accuracy"] >= target_selective_accuracy and c["has_both_classes"]]
    if feasible:
        # Maximize coverage; then narrower ambiguous band.
        feasible.sort(key=lambda c:(c["coverage"], c["selective_accuracy"], -(c["high"]-c["low"])), reverse=True)
        best=feasible[0]
        best["selection_rule"]="max_coverage_subject_to_selective_accuracy>=0.95_and_both_classes"
    else:
        candidates.sort(key=lambda c:(c["selective_accuracy"], c["coverage"]), reverse=True)
        best=candidates[0]
        best["selection_rule"]="fallback_max_selective_accuracy_then_coverage"
    return best, candidates


def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    train=pd.read_csv(DATA/'dev_train.csv')
    val=pd.read_csv(DATA/'dev_validation.csv')
    diag=pd.read_csv(DATA/'dev_scenario_test.csv')

    model=build_model()
    model.fit(train['text'],train['label'])

    # Scenario-aware diagnostic test: entire source templates unseen during training.
    diag_pred=model.predict(diag['text'])
    diag_prob=model.predict_proba(diag['text'])[:,1]
    diag_metrics=metrics(diag['label'].values,diag_pred)
    diag_out=diag[['text','label','scenario_id','category']].copy()
    diag_out['pred_label']=diag_pred
    diag_out['scam_probability']=diag_prob
    diag_out.to_csv(RESULTS/'lr_scenario_test_predictions.csv',index=False,encoding='utf-8-sig')

    # Gate calibration uses validation only.
    val_prob=model.predict_proba(val['text'])[:,1]
    gate,_=choose_gate(val['label'].values.astype(int),val_prob)

    # Binary validation is reported only as development diagnostic.
    val_pred=(val_prob >= 0.5).astype(int)
    val_metrics=metrics(val['label'].values,val_pred)

    # Freeze final LR using train+validation. Scenario diagnostic test is still excluded.
    final_train=pd.concat([train,val],ignore_index=True)
    final_model=build_model()
    final_model.fit(final_train['text'],final_train['label'])
    joblib.dump(final_model,MODELS/'safenet_lr_research_v1.joblib')

    calibration={
        "version":"lr_gate_v1",
        "frozen_on":"2026-10-02",
        "selection_data":"dev_validation.csv only",
        "benign_threshold":gate['low'],
        "scam_threshold":gate['high'],
        "validation_gate":gate,
        "binary_threshold":0.5,
        "note":"Thresholds are frozen before independent_test_v1 evaluation.",
    }
    (MODELS/'lr_calibration_v1.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2),encoding='utf-8')

    report={
        "development_only":True,
        "train_rows":len(train),
        "validation_rows":len(val),
        "scenario_test_rows":len(diag),
        "validation_binary_metrics":val_metrics,
        "scenario_aware_diagnostic_metrics":diag_metrics,
        "gate_calibration":calibration,
        "final_lr_training_rows":len(final_train),
        "final_lr_training_splits":["train","validation"],
        "independent_test_used":False,
    }
    (RESULTS/'lr_training_calibration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Saved:')
    print(' ',MODELS/'safenet_lr_research_v1.joblib')
    print(' ',MODELS/'lr_calibration_v1.json')


if __name__=='__main__':
    main()

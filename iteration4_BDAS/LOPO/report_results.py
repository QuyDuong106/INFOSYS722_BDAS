"""Verify and report only completed Research Paper nested Optuna LOPO outputs.

No resampling, new scores, tuning or training is performed. Original metrics
are checked against original held-out probabilities before generating the table.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

ROOT=Path(__file__).resolve().parent
NAMES={'logistic':'Logistic regression','random_forest':'Random forest',
       'xgboost':'XGBoost','lightgbm':'LightGBM','stacking':'Stacking'}
METRICS=['roc_auc','accuracy','precision','recall','specificity','f1']


def build():
    collected=[];predictions=[];sources={}
    for model in NAMES:
        folder=ROOT/'results'/f'optuna_{model}'
        config=json.loads((folder/'run_config.json').read_text())
        progress=json.loads((folder/'progress.json').read_text())
        if not (progress['complete'] and progress['evaluated_folds']==319
                and config['inner_folds']==3 and config['trials']==20 and config['seed']==42):
            raise ValueError(f'Incomplete or incompatible observed run: {model}')
        pred=pd.read_csv(folder/'predictions.csv');saved=pd.read_csv(folder/'metrics.csv')
        if len(pred)!=319 or not pred.source_row.is_unique or len(saved)!=1:
            raise ValueError(f'Require 319 unique excluded-patient predictions: {model}')
        if not pred.model.eq(model).all() or not pred.probability_present.between(0,1).all():
            raise ValueError(f'Invalid observed prediction evidence: {model}')
        y=pred.actual_present.to_numpy();prob=pred.probability_present.to_numpy();label=prob>=.5
        tn,fp,fn,tp=confusion_matrix(y,label,labels=[0,1]).ravel()
        measured=dict(roc_auc=roc_auc_score(y,prob),accuracy=accuracy_score(y,label),
            precision=precision_score(y,label,zero_division=0),recall=recall_score(y,label),
            specificity=tn/(tn+fp),f1=f1_score(y,label),TN=tn,FP=fp,FN=fn,TP=tp)
        for metric,value in measured.items():
            if not np.isclose(float(saved.iloc[0][metric]),value,rtol=0,atol=1e-12):
                raise ValueError(f'Saved metric differs from predictions: {model} {metric}')
        # Copy the observed metrics, rather than replace them with another estimate.
        row=saved.iloc[0]
        collected.append(dict(model=model,seed=int(row.seed),rows=int(row.rows),
            **{key:float(row[key]) for key in METRICS},
            **{key:int(row[key]) for key in ['TN','FP','FN','TP']}))
        predictions.append(pred)
        sources[model]={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                        for name in ['metrics.csv','predictions.csv','run_config.json','progress.json']}
    table=pd.DataFrame(collected)
    table.to_csv(ROOT/'results/metrics_combined.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(ROOT/'results/predictions_combined.csv',index=False)
    (ROOT/'results/report_validation.json').write_text(json.dumps(dict(
        source='Observed completed Research Paper outputs',training_seed=42,
        outer_folds=319,training_records_per_outer_fit=318,inner_cv_folds=3,
        optuna_trials_per_model_per_outer_fit=20,bootstrap_applied=False,
        new_training_performed=False,metric_verification='all saved scores match held-out probabilities to 1e-12',
        source_hashes=sources),indent=2))
    rows=[]
    for row in collected:
        values=[]
        for metric in METRICS:
            value=f'{row[metric]:.3f}'
            if round(row[metric],3)==round(table[metric].max(),3):value=r'\textbf{'+value+'}'
            values.append(value)
        rows.append(' & '.join([NAMES[row['model']],*values])+r' \\')
    content=r'''\begin{table}[H]
\centering\small
\caption{Observed Research Paper results: nested Optuna LOPO with three-fold training CV, seed 42. Scores pool all 319 held-out patients; each outer model trains on 318 records. Highest displayed scores are bold. No bootstrap estimates are included.}
\label{tab:research-optuna-lopo}
\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrr}
\toprule
Model & AUC & Accuracy & Precision & Recall & Specificity & F1 \\
\midrule
'''+ '\n'.join(rows)+'\n'+r'''\bottomrule
\end{tabular}}
\end{table}
'''
    (ROOT.parent/'Report/generated/research_optuna_lopo.tex').write_text(content)
    return table

if __name__=='__main__':print(build().to_string(index=False))

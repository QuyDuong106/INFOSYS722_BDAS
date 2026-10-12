"""Repeated 70/30 stratified holdout, with training-only three-fold Optuna CV."""
import argparse,json,hashlib
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits
from optuna_holdout import tune_training_fold
from utils import *
VARIANTS={'cart_yeo_standard':('cart','yeo_standard'),'cart_standard':('cart','standard'),'rfe_yeo_standard':('rfe','yeo_standard'),'rfe_standard':('rfe','standard')}
def resolve_output_dir(config, output_dir=None):
 """Keep incompatible runtime results separate; never mix checkpoints."""
 default=ROOT/'results/holdout'
 out=Path(output_dir) if output_dir is not None else default
 cp=out/'run_config.json'
 if cp.exists() and json.loads(cp.read_text())!=config:
  previous=json.loads(cp.read_text())
  changed=[key for key in sorted(set(previous)|set(config)) if previous.get(key)!=config.get(key)]
  if output_dir is not None:
   raise ValueError(f"Output folder {out} has different configuration fields: {', '.join(changed)}. Use output_dir=None for automatic separation, or choose a fresh folder.")
  digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
  out=ROOT/'results/holdout_runs'/digest
  print(f"Saved run differs in {', '.join(changed)}; using {out}",flush=True)
  alternate=out/'run_config.json'
  if alternate.exists() and json.loads(alternate.read_text())!=config:
   raise ValueError(f"Configuration collision at {out}; choose a fresh explicit output_dir")
 return out

def run(trials=20,seeds=DEFAULT_SEEDS,models=WORKFLOW_MODELS,output_dir=None,variants=None):
 if len(seeds)!=5 or len(set(seeds))!=5 or trials<1:raise ValueError('Require five distinct seeds and positive trials')
 all_variants={**VARIANTS,'rfe10_standard':('rfe','standard')}
 chosen=VARIANTS if variants is None else {v:all_variants[v] for v in variants}
 if not chosen:raise ValueError('Choose at least one preparation variant')
 X,y=workflow_data()
 config=dict(version=1,signature=frame_signature(X,y),seeds=list(seeds),models=list(models),trials=trials,test_fraction=.3,inner_folds=3,top_k=20,threshold=.5,tuned_variant='rfe_standard',packages={p:version(p) for p in ['optuna','scikit-learn','xgboost','lightgbm','numpy','pandas']})
 if variants is not None and list(variants)==['rfe10_standard']:config['top_k']=10
 if variants is not None:config['variants']={v:dict(selection=pair[0],scaling=pair[1],top_k=10 if v=='rfe10_standard' else 20) for v,pair in chosen.items()}
 out=resolve_output_dir(config,output_dir);out.mkdir(parents=True,exist_ok=True)
 print(f'Results directory: {out}',flush=True)
 cp=out/'run_config.json'
 cp.write_text(json.dumps(config,indent=2));rows=[];predictions=[]
 for seed in seeds:
  train,test=train_test_split(np.arange(len(X)),test_size=.3,stratify=y,random_state=seed);assert not set(train)&set(test)
  xt,yt=X.iloc[train],y.iloc[train];xv,yv=X.iloc[test],y.iloc[test]
  sd=out/f'seed_{seed}';sd.mkdir(exist_ok=True)
  pd.concat([pd.DataFrame({'source_row':X.iloc[pos].index,'partition':label}) for label,pos in [('train',train),('test',test)]]).to_csv(sd/'split_membership.csv',index=False)
  for variant,(selection,scaling) in chosen.items():
   top_k=10 if variant=='rfe10_standard' else 20
   for mode in (('benchmark','tuned') if variant=='rfe_standard' else ('benchmark',)):
    for name in models:
     dest=sd/variant/mode/name;dest.mkdir(parents=True,exist_ok=True);checkpoint=dest/'metrics.json'
     if checkpoint.exists():row=json.loads(checkpoint.read_text());pred=pd.read_csv(dest/'predictions.csv')
     else:
      print(f'Seed {seed}: {variant} {mode} {name}',flush=True);started=perf_counter()
      with threadpool_limits(limits=1):
       if mode=='tuned':
        fitted,params,score,trial_table=tune_training_fold(name,xt,yt,seed=seed,trials=trials,selection=selection,scaling=scaling,top_k=top_k)
        trial_table.to_csv(dest/'optuna_trials.csv',index=False)
       else:
        fitted=DummyClassifier(strategy='prior') if name=='dummy_prior' else make_evaluation_model(name,xt,seed,selection,top_k,scaling=scaling)
        score=float(cross_val_score(fitted,xt,yt,cv=StratifiedKFold(3,shuffle=True,random_state=seed),scoring='roc_auc',n_jobs=1).mean());params={};fitted.fit(xt,yt)
       probability=disease_probability(fitted,xv)
      row=metric_row(seed,name,mode,yv,probability,perf_counter()-started);row.update(variant=variant,inner_cv_auc=score,auc_benchmark_met=row['roc_auc']>=.8,sensitivity_benchmark_met=row['recall']>=.8,fnr=1-row['recall'])
      pred=prediction_rows(seed,name,mode,xv,yv,probability);pred['variant']=variant
      dump_pickle(fitted,dest/'model.pkl');pred.to_csv(dest/'predictions.csv',index=False)
      if name!='dummy_prior':selected_feature_table(fitted).to_csv(dest/'selected_features.csv',index=False)
      (dest/'best_parameters.json').write_text(json.dumps(dict(parameters=params,inner_cv_auc=score),indent=2));checkpoint.write_text(json.dumps(row,indent=2,allow_nan=False))
     rows.append(row);predictions.append(pred)
 metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics_per_seed.csv',index=False);pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
 values=['roc_auc','accuracy','recall','specificity','precision','f1','log_loss','inner_cv_auc','fnr','fit_predict_seconds']
 summary=metrics.groupby(['variant','evaluation','model'])[values].agg(['mean','std','min','max']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary.csv',index=False)
 for mode in ['benchmark','tuned']:metrics[metrics.evaluation.eq(mode)].to_csv(out/f'{mode}_metrics.csv',index=False)
 paired=metrics[metrics.evaluation.eq('tuned')].merge(metrics[metrics.evaluation.eq('benchmark') & metrics.variant.eq('rfe_standard')],on=['seed','model'],suffixes=('_tuned','_benchmark'))
 for v in values:paired[v+'_delta']=paired[v+'_tuned']-paired[v+'_benchmark']
 paired.to_csv(out/'tuning_comparison.csv',index=False)
 selected=metrics[metrics.evaluation.eq('tuned') & ~metrics.model.eq('dummy_prior')].sort_values(['seed','inner_cv_auc','model'],ascending=[True,False,True]).groupby('seed',sort=False).head(1)
 selected.to_csv(out/'training_cv_selected_test_metrics.csv',index=False)
 (out/'complete.json').write_text(json.dumps(dict(complete=True,model_evaluations=len(metrics),seeds=list(seeds))))
 metrics.attrs['output_dir']=str(out.resolve())
 return metrics
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=20);p.add_argument('--output-dir',type=Path);p.add_argument('--seeds',type=int,nargs=5,default=list(DEFAULT_SEEDS));p.add_argument('--models',nargs='+',choices=WORKFLOW_MODELS,default=list(WORKFLOW_MODELS));p.add_argument('--variants',nargs='+',choices=[*VARIANTS,'rfe10_standard']);run(**vars(p.parse_args()))

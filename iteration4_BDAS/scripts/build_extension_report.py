"""Report measured saved-Colab Spark results and matched Python RFE10 evidence."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spark-results',type=Path);args=p.parse_args()
gen=R/'Report/generated';gen.mkdir(exist_ok=True)
names={'logistic':'Logistic regression','random_forest':'Random forest','gbt':'Spark GBT','xgboost':'XGBoost','lightgbm':'LightGBM','stacking':'Stacking'}
def table(caption,headers,rows,label):
 return '\\begin{table}[H]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
def ms(mean,std,bold=False):
 value=f'{mean:.3f}'
 if bold:value=r'\textbf{'+value+'}'
 return f'{value} $\\pm$ {std:.3f}'
spark=pd.read_csv(R/'results/spark_colab_reference/metrics_per_seed.csv')
assert len(spark)==15 and spark.groupby('model').seed.nunique().eq(5).all()
summary=spark.groupby('model')[['roc_auc','accuracy','sensitivity','specificity']].agg(['mean','std'])
rows=[]
for model,row in summary.iterrows():
 rows.append([names[model],*[ms(row[(metric,'mean')],row[(metric,'std')],np.isclose(row[(metric,'mean')],summary[(metric,'mean')].max(),rtol=0,atol=1e-12)) for metric in ['roc_auc','sensitivity','specificity','accuracy']]])
text=table('Measured native Spark baseline from the author-saved Colab notebook: mean and sample SD over five test holdouts. Highest means are bold.','Model|AUC|Sensitivity|Specificity|Accuracy'.split('|'),rows,'tab:spark-colab-execution')
text+='Each model has five 96-record test evaluations. These scores were recovered from the saved six-decimal display, not from a new local run. The three native Spark families and all-field preparation differ from the five-family Python prototypes; their scores are not pooled together.\n'
(gen/'spark_execution_evidence.tex').write_text(text)
fig,axes=plt.subplots(1,3,figsize=(11,3.6));vmax=spark[['TN','FP','FN','TP']].max().max()
for ax,(model,g) in zip(axes,spark.groupby('model')):
 vals=g[['TN','FP','FN','TP']].mean().to_numpy().reshape(2,2)
 ax.imshow(vals,cmap='Blues',vmin=0,vmax=vmax)
 for (i,j),value in np.ndenumerate(vals):ax.text(j,i,f'{value:.1f}',ha='center',va='center')
 ax.set(title=names[model],xticks=[0,1],yticks=[0,1],xticklabels=['Absent','Present'],yticklabels=['Absent','Present'],xlabel='Predicted',ylabel='Actual')
fig.tight_layout();dest=R/'Report/figures/modeling';dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/'spark_average_cm.pdf');fig.savefig(dest/'spark_average_cm.png',dpi=180)
ten=pd.read_csv(R/'results/holdout_rfe10/metrics_per_seed.csv');base=pd.read_csv(R/'results/holdout/metrics_per_seed.csv')
base=base[base.variant.eq('rfe_standard')&base.evaluation.eq('benchmark')&~base.model.eq('dummy_prior')]
ten=ten[~ten.model.eq('dummy_prior')];paired=base.merge(ten,on=['seed','model'],suffixes=('_20','_10'),validate='one_to_one');assert len(paired)==25
rows=[]
for model,g in paired.groupby('model'):
 rows.append([names[model],ms(g.roc_auc_20.mean(),g.roc_auc_20.std()),ms(g.roc_auc_10.mean(),g.roc_auc_10.std()),f'{(g.roc_auc_10-g.roc_auc_20).mean():+.3f}',f'{0.0 if abs((g.recall_10-g.recall_20).mean())<1e-12 else (g.recall_10-g.recall_20).mean():+.3f}'])
text=table('Completed Python prototype feature-count iteration: fixed settings and matched five-seed holdouts, RFE10 minus RFE20. Both use standardization.','Model|RFE20 AUC|RFE10 AUC|$\\Delta$ AUC|$\\Delta$ sensitivity'.split('|'),rows,'tab:rfe-feature-count')
text+='This is a controlled feature-count comparison within the Python implementation. It does not supply native Spark CART/RFE scores. All saved selectors and stacking base pipelines retain exactly ten fields in the new run.\n'
(gen/'feature_count_evidence.tex').write_text(text)
paired.to_csv(R/'results/holdout_rfe10/feature_count_comparison.csv',index=False)
text='Native Spark CART20, RFE20 and RFE10 are implemented in the new pipeline extension. The submitted evidence currently includes the measured all-field Colab baseline; additional native feature-selection scores will be inserted only after a complete Spark run.\n'
if args.spark_results:
 directory=args.spark_results;progress=json.loads((directory/'progress.json').read_text());assert progress['complete']
 m=pd.DataFrame(json.loads((directory/'metrics.json').read_text()));assert m.groupby(['variant','model']).seed.nunique().eq(5).all()
 rows=[]
 for (variant,model),g in m.groupby(['variant','model']):
  rows.append([variant.replace('_',r'\_'),names[model],str(int(g.retained_features.iloc[0])),ms(g.roc_auc.mean(),g.roc_auc.std()),ms(g.sensitivity.mean(),g.sensitivity.std())])
 text=table('Completed native Spark feature-selection iterations: five matched holdouts per family and preparation version.','Version|Model|Fields|AUC|Sensitivity'.split('|'),rows,'tab:spark-selection-iterations')
 text+='Source: a completed Spark extension output with recorded data/code signatures, CV search evidence and selected-field artifacts.\n'
(gen/'spark_iteration_evidence.tex').write_text(text)
print('Generated measured Spark baseline, RFE10/RFE20 comparison, and honest native extension status.')

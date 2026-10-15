"""Generate report tables and averaged confusion matrices from completed runs."""
from pathlib import Path
from report_table_format import bold_best_cells
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir',type=Path,default=R/'results/holdout')
args=parser.parse_args()
out=args.results_dir;gen=R/'Report/generated';gen.mkdir(exist_ok=True)
if not json.loads((out/'complete.json').read_text())['complete']:raise RuntimeError('Run all five seeds first')
m=pd.read_csv(out/'metrics_per_seed.csv');s=pd.read_csv(out/'summary.csv');primary=s[s.variant.eq('rfe_standard')]
names={'logistic':'Logistic regression','random_forest':'Random forest','xgboost':'XGBoost','lightgbm':'LightGBM','stacking':'Stacking','dummy_prior':'Prior dummy'}
def table(caption,headers,rows,label):
 rows=bold_best_cells(rows,headers)
 return '\\begin{table}[H]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+ 'l'+'r'*(len(headers)-1)+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+ '\n\\midrule\n'+'\n'.join(' & '.join(str(x) for x in row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
def ms(row,metric,bold=False):
 mean=f"{row[metric+'_mean']:.3f}"
 if bold:mean=r"\textbf{"+mean+"}"
 return f"{mean} $\\pm$ {row[metric+'_std']:.3f}"
rows=[]
for name in names:
 b=primary[primary.model.eq(name)&primary.evaluation.eq('benchmark')].iloc[0];t=primary[primary.model.eq(name)&primary.evaluation.eq('tuned')].iloc[0]
 rows.append([names[name],ms(b,'inner_cv_auc'),ms(t,'inner_cv_auc') if name!='dummy_prior' else 'Not tuned'])
winners=m[m.evaluation.eq('tuned')&~m.model.eq('dummy_prior')].sort_values('inner_cv_auc',ascending=False).groupby('seed',sort=False).first()
text=table('Algorithm exploration: training-only three-fold CV AUC, mean and sample SD across five seeds.','Model|Fixed CV AUC|Best-trial CV AUC'.split('|'),rows,'tab:holdout-selection')
text+='Training-CV choices are '+', '.join(f"seed {seed}: {names[row.model]} ({row.inner_cv_auc:.3f})" for seed,row in winners.sort_index().iterrows())+'. These are best-trial training scores, not test scores.\n'
(gen/'holdout_selection.tex').write_text(text)
rows=[]
execution=primary[~(primary.model.eq('dummy_prior') & primary.evaluation.eq('tuned'))]
execution_metrics=('roc_auc','recall','specificity','accuracy')
maxima={metric:execution[metric+'_mean'].max() for metric in execution_metrics}
for _,row in execution.iterrows():
 rows.append([names[row.model]+' ('+row.evaluation+')',*[ms(row,metric,bold=np.isclose(row[metric+'_mean'],maxima[metric],rtol=0,atol=1e-12)) for metric in execution_metrics]])
text=table('RFE20/standardization test results: mean and sample SD across five holdouts. Highest column means are bold, including the prior dummy; exact ties share highlighting.','Model|AUC|Sensitivity|Specificity|Accuracy'.split('|'),rows,'tab:holdout-results')
tuned=m[m.evaluation.eq('tuned')&~m.model.eq('dummy_prior')]
text+='Both project targets were met in '+', '.join(f"{names[n]}: {int((g.auc_benchmark_met & g.sensitivity_benchmark_met).sum())}/5 splits" for n,g in tuned.groupby('model'))+'. Full metric distributions and all four benchmark variants are saved in the result CSVs.\n'
(gen/'holdout_results.tex').write_text(text)

# Dedicated held-out tables keep tuned models prominent in the execution section.
ordered=['logistic','random_forest','xgboost','lightgbm','stacking']
optuna=primary[primary.evaluation.eq('tuned') & primary.model.isin(ordered)]
assert len(optuna)==5
raw=m[m.variant.eq('rfe_standard') & m.evaluation.eq('tuned') & m.model.isin(ordered)]
assert len(raw)==25 and raw.groupby('model').seed.nunique().eq(5).all()
rows=[]
for name in ordered:
 row=optuna[optuna.model.eq(name)].iloc[0]
 rows.append([names[name],*[ms(row,metric) for metric in ['roc_auc','accuracy','precision','recall','f1']]])
text=table('Completed non-Spark Optuna-tuned holdout results: mean and sample SD across five matched seeds, RFE20 and standardization. Highest displayed means are bold.', 'Model|AUC|Accuracy|Precision|Recall|F1'.split('|'),rows,'tab:holdout-optuna-test')
(gen/'holdout_optuna_results.tex').write_text(text)
rows=[]
for name in [*ordered,'dummy_prior']:
 row=primary[primary.model.eq(name)&primary.evaluation.eq('benchmark')].iloc[0]
 rows.append([names[name],*[ms(row,metric) for metric in ['roc_auc','recall','specificity','accuracy']]])
text=table('Non-Spark fixed-parameter holdout reference: RFE20 and standardization with the same five test partitions. Highest displayed means are bold, including ties and the prior dummy.', 'Model|AUC|Recall|Specificity|Accuracy'.split('|'),rows,'tab:holdout-fixed-test')
(gen/'holdout_benchmark_results.tex').write_text(text)

rows=[]
for n in names:
 if n=='dummy_prior':continue
 means=s[s.evaluation.eq('benchmark')&s.model.eq(n)].set_index('variant').roc_auc_mean
 rows.append([names[n],*[f'{means[v]:.3f}' for v in ['cart_yeo_standard','cart_standard','rfe_yeo_standard','rfe_standard']]])
text=table('Controlled fixed-model preparation comparisons: mean test AUC over five matched seeds.','Model|CART+YJ|CART+Std|RFE+YJ|RFE+Std'.split('|'),rows,'tab:holdout-preparation')
(gen/'holdout_preparation.tex').write_text(text)
p=pd.read_csv(out/'tuning_comparison.csv');rows=[]
for n,g in p[~p.model.eq('dummy_prior')].groupby('model'):
 rows.append([names[n],f'{g.roc_auc_delta.mean():+.3f}',f'{g.recall_delta.mean():+.3f}',f'{int((g.roc_auc_delta>0).sum())}/5'])
text+=table('Matched tuning effects on RFE/standardization: tuned minus fixed benchmark.','Model|Mean AUC change|Mean sensitivity change|AUC gains'.split('|'),rows,'tab:holdout-tuning')
(gen/'holdout_iterations.tex').write_text(text)
fig,axes=plt.subplots(2,3,figsize=(11,6.7));vmax=max(tuned.groupby('model')[['TN','FP','FN','TP']].mean().max())
for ax,(n,g) in zip(axes.flat,tuned.groupby('model')):
 vals=g[['TN','FP','FN','TP']].mean().to_numpy().reshape(2,2);ax.imshow(vals,cmap='Blues',vmin=0,vmax=vmax)
 for (i,j),value in np.ndenumerate(vals):ax.text(j,i,f'{value:.1f}',ha='center',va='center')
 ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Absent','Present'],yticklabels=['Absent','Present'],xlabel='Predicted',ylabel='Actual',title=names[n])
axes.flat[-1].axis('off');fig.tight_layout();dest=R/'Report/figures/modeling';dest.mkdir(parents=True,exist_ok=True)
fig.savefig(dest/'holdout_average_cm.pdf');fig.savefig(dest/'holdout_average_cm.png',dpi=180)
print('Generated holdout report evidence and averaged confusion matrices.')

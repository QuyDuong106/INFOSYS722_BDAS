import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
import holdout_workflow as h
from utils import workflow_data,load_pickle
class HoldoutTests(unittest.TestCase):
 def test_training_only_search_and_exact_five_seed_outputs(self):
  X=pd.DataFrame(np.arange(200).reshape(40,5),columns=list('abcde'));y=pd.Series([0,1]*20)
  seen=[]
  def tune(name,xt,yt,**kwargs):
   seen.append((kwargs['seed'],set(xt.index)))
   return DummyClassifier(strategy='prior').fit(xt,yt),{},.5,pd.DataFrame()
  with tempfile.TemporaryDirectory() as tmp,patch.object(h,'workflow_data',return_value=(X,y)),patch.object(h,'tune_training_fold',side_effect=tune),patch.object(h,'VARIANTS',{'rfe_standard':('rfe','standard')}):
   metrics=h.run(trials=1,models=('dummy_prior',),output_dir=tmp)
   self.assertEqual(len(metrics),10);self.assertEqual(len(seen),5)
   for seed,training in seen:
    membership=pd.read_csv(Path(tmp)/f'seed_{seed}/split_membership.csv');test=set(membership.loc[membership.partition.eq('test'),'source_row'])
    self.assertFalse(training & test);self.assertEqual(len(training),28);self.assertEqual(len(test),12)
   # Identical resume should not execute the search again.
   h.run(trials=1,models=('dummy_prior',),output_dir=tmp);self.assertEqual(len(seen),5)
   with self.assertRaises(ValueError):h.run(trials=2,models=('dummy_prior',),output_dir=tmp)
 def test_environment_mismatch_uses_separate_stable_default(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(h,'ROOT',Path(tmp)):
   original={'trials':20,'packages':{'optuna':'local'}}
   current={'trials':20,'packages':{'optuna':'colab'}}
   base=Path(tmp)/'results/holdout';base.mkdir(parents=True)
   (base/'run_config.json').write_text(json.dumps(original))
   alternative=h.resolve_output_dir(current)
   self.assertNotEqual(alternative,base)
   self.assertEqual(alternative.parent,Path(tmp)/'results/holdout_runs')
   alternative.mkdir(parents=True);(alternative/'run_config.json').write_text(json.dumps(current))
   self.assertEqual(h.resolve_output_dir(current),alternative)
   self.assertEqual(json.loads((base/'run_config.json').read_text()),original)
   self.assertEqual(h.resolve_output_dir(original),base)
   with self.assertRaisesRegex(ValueError,'packages'):
    h.resolve_output_dir(current,base)
 def test_source_labels_and_split_sizes(self):
  from sklearn.model_selection import train_test_split
  X,y=workflow_data();self.assertEqual(len(X),319);self.assertEqual(int(y.sum()),161)
  for seed in h.DEFAULT_SEEDS:
   train,test=train_test_split(X.index,test_size=.3,stratify=y,random_state=seed)
   self.assertEqual(len(train),223);self.assertEqual(len(test),96);self.assertFalse(set(train)&set(test))
if __name__=='__main__':unittest.main()

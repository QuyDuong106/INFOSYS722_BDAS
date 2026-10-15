"""CPU-verifiable GPU search/resume safeguards; no real research run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from spark_lopo_acceleration import check_cuda, tune_xgboost

class AccelerationTests(unittest.TestCase):
    def test_cuda_fallback_is_rejected(self):
        class CPUFallback:
            def save_config(self):
                return json.dumps({'learner':{'generic_param':{'device':'cpu'}}})
        with patch('xgboost.train',return_value=CPUFallback()):
            with self.assertRaisesRegex(RuntimeError,'did not use CUDA'):
                check_cuda()

    def test_optuna_cpu_reference_and_completed_trials_resume(self):
        import xgboost
        rng=np.random.default_rng(42)
        x=rng.normal(size=(36,3)).astype(np.float32)
        y=np.tile([0,1],18).astype(np.float32)
        # Each validation fold contains both classes; arrays are already prepared.
        pairs=[((x[np.arange(36)%3!=i],y[np.arange(36)%3!=i]),
                (x[np.arange(36)%3==i],y[np.arange(36)%3==i])) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            best,score,rows,base=tune_xgboost(pairs,seed=42,trials=2,directory=tmp,use_gpu=False)
            self.assertEqual(len(rows),2)
            self.assertEqual(base['device'],'cpu')
            self.assertTrue(0<=score<=1)
            self.assertTrue(all(len(row['fold_auc'])==3 for row in rows))
            with patch('xgboost.train',side_effect=AssertionError('Completed trials must not refit')):
                again=tune_xgboost(pairs,seed=42,trials=2,directory=tmp,use_gpu=False)
            self.assertEqual(best,again[0]);self.assertEqual(score,again[1])
            # Resume one additional trial rather than restarting the study.
            _,_,resumed,_=tune_xgboost(pairs,seed=42,trials=3,directory=tmp,use_gpu=False)
            self.assertEqual(len(resumed),3)
            self.assertEqual(rows,resumed[:2])

if __name__=='__main__':unittest.main()

import os
import unittest

import numpy as np
import pandas as pd

from batchprocessing import batchprocessing

TEMPDIR = os.path.join('tests', 'temp')

class MyMonkeyPatch:
    def __init__(self):
        self.colnames = ['A', 'B']
        self.df = pd.DataFrame(np.zeros((100, 2)))
        self.rnd_df = pd.DataFrame(np.random.randn(100,2))
        self.df.columns = self.colnames
        self.rnd_df.columns = self.colnames
        self.checkpoint_path=os.path.join(TEMPDIR, 'mytest')
        self.fake_cp_path = os.path.join(TEMPDIR, 'mytest_fake')

    @batchprocessing.batch_predict
    def add(self, X, n_batches=None, checkpoint_path=None):
        return X + 1


class TestBatchProc(unittest.TestCase):
    def _setup(self):
        self.myobj = MyMonkeyPatch()
        batchprocessing._check_makedir(self.myobj.fake_cp_path)
        splits = np.array_split(self.myobj.rnd_df, 10)
        for i in range(5):
            splits_df = pd.DataFrame(splits[i])
            batchprocessing._save_checkpoints(
                self.myobj.fake_cp_path,
                iteration=i,
                df=splits_df
            )

    def test_batch_predict(self):
        self._setup()
        x = self.myobj.add(X=self.myobj.df,
                           n_batches=10,
                           checkpoint_path=self.myobj.checkpoint_path)
        assertion_df = pd.DataFrame(np.ones((100, 2)))
        assertion_df.columns = self.myobj.colnames
        assert x.shape == (100, 2)
        assert x.equals(assertion_df)

    def test_check_makedir(self):
        self._setup()
        batchprocessing._check_makedir(self.myobj.checkpoint_path)
        assert os.path.isdir(self.myobj.checkpoint_path)

    def test_cleanup_checkpoints(self):
        self._setup()
        batchprocessing._check_makedir(self.myobj.checkpoint_path)
        batchprocessing._cleanup_checkpoints(self.myobj.checkpoint_path)
        assert os.path.isdir(self.myobj.checkpoint_path) is False

    def test_load_checkpoints(self):
        self._setup()
        x = self.myobj.add(X=self.myobj.df,
                           n_batches=10,
                           checkpoint_path=self.myobj.fake_cp_path)
        assertion_df1 = pd.DataFrame(np.ones((100, 2)))
        assertion_df1.columns = self.myobj.colnames
        assertion_df2 = pd.concat(
            [self.myobj.rnd_df.iloc[:50,:],
            assertion_df1.iloc[50:,:]],
            axis=0, ignore_index=True
        )
        assert x.shape == (100, 2)
        assert x.equals(assertion_df1) is False
        assert np.allclose(x, assertion_df2, rtol=0.0001)

if __name__ == '__main__':
    unittest.main()

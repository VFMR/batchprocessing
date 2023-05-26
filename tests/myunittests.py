import os
import unittest
import time

import numpy as np
import pandas as pd

from batchprocessing import batchprocessing

TEMPDIR = os.path.join('tests', 'temp')


class MyMonkeyPatch:
    def __init__(self):
        self.colnames = ['A', 'B']
        self.df = pd.DataFrame(np.zeros((100, 2)))
        self.rnd_df = pd.DataFrame(np.random.randn(100, 2))
        self.df.columns = self.colnames
        self.rnd_df.columns = self.colnames
        self.checkpoint_path=os.path.join(TEMPDIR, 'mytest')
        self.fake_cp_path = os.path.join(TEMPDIR, 'mytest_fake')
        self._n_batches = 10

    @batchprocessing.BatchProcessor.batch_predict_auto
    def add(self, X, n_batches=None, checkpoint_path=None, n_jobs=1):
        time.sleep(.2)
        return X + 1


processor = batchprocessing.BatchProcessor(
        n_batches=1,
        checkpoint_path=os.path.join(TEMPDIR, 'mytest'),
        n_jobs=1,
        do_load_cp=False,
        )


processor2 = batchprocessing.BatchProcessor(
        n_batches=10,
        checkpoint_path=os.path.join(TEMPDIR, 'mytest'),
        n_jobs=2,
        do_load_cp=False,
        )

@processor.batch_predict
def my_func(X):
    return X + 1


def test_classbased_bp():
    X1 = pd.DataFrame(np.zeros((100, 2)))
    X2 = pd.DataFrame(np.arange(100))
    result = my_func(X=X1)
    assert np.allclose(result, pd.DataFrame(np.ones((100, 2))))
    result = my_func(X=X2)
    assert np.allclose(result, pd.DataFrame(np.arange(100))+1)


@processor2.batch_predict
def my_func2(X):
    return X + 1


def test_classbased_bp():
    X1 = pd.DataFrame(np.zeros((100, 2)))
    X2 = pd.DataFrame(np.arange(100))
    result = my_func2(X=X1)
    assert np.allclose(result, pd.DataFrame(np.ones((100, 2))))
    result = my_func2(X=X2)
    assert np.allclose(result, pd.DataFrame(np.arange(100))+1)


class TestBatchProc(unittest.TestCase):
    def _setup(self):
        self.myobj = MyMonkeyPatch()
        self._n_batches = self.myobj._n_batches
        splits = np.array_split(self.myobj.rnd_df, self._n_batches)
        processor = batchprocessing.BatchProcessor(
                n_batches=self._n_batches,
                checkpoint_path=self.myobj.fake_cp_path)
        processor._check_makedir()
        # only save the first 5 iterations
        for i in range(5):
            splits_df = pd.DataFrame(splits[i])
            processor._save_checkpoints(
                iteration=i,
                df=splits_df,
            )

    def test_batch_predict(self):
        self._setup()
        x = self.myobj.add(X=self.myobj.df,
                           n_batches=self._n_batches,
                           checkpoint_path=self.myobj.checkpoint_path)
        assertion_df = pd.DataFrame(np.ones((100, 2)))
        assertion_df.columns = self.myobj.colnames
        assert x.shape == (100, 2)
        assert x.equals(assertion_df)

    def test_batch_predict_parallel(self):
        self._setup()
        x = self.myobj.add(X=self.myobj.df,
                           n_batches=self._n_batches,
                           checkpoint_path=self.myobj.checkpoint_path,
                           n_jobs=4)
        assertion_df = pd.DataFrame(np.ones((100, 2)))
        assertion_df.columns = self.myobj.colnames
        assert x.shape == (100, 2)
        assert x.equals(assertion_df)

    def test_check_makedir(self):
        self._setup()
        processor = batchprocessing.BatchProcessor(
                n_batches=self._n_batches,
                checkpoint_path=self.myobj.checkpoint_path)
        processor._check_makedir()
        assert os.path.isdir(self.myobj.checkpoint_path)

    def test_cleanup_checkpoints(self):
        self._setup()
        processor = batchprocessing.BatchProcessor(
                n_batches=self._n_batches,
                checkpoint_path=self.myobj.checkpoint_path)
        processor._check_makedir()
        processor._cleanup_checkpoints()
        assert os.path.isdir(self.myobj.checkpoint_path) is False

    # def test_load_checkpoints(self):
    #     self._setup()
    #     x = self.myobj.add(X=self.myobj.df,
    #                        n_batches=self._n_batches,
    #                        checkpoint_path=self.myobj.fake_cp_path)
    #     assertion_df1 = pd.DataFrame(np.ones((100, 2)))
    #     assertion_df1.columns = self.myobj.colnames
    #     assertion_df2 = pd.concat(
    #         [self.myobj.rnd_df.iloc[:50, :], assertion_df1.iloc[50:, :]],
    #         axis=0,
    #         ignore_index=True
    #     )
    #     print(x)
    #     print()
    #     print(assertion_df2)
    #     assert x.shape == (100, 2)
    #     assert x.equals(assertion_df1) is False
    #     assert np.allclose(x, assertion_df2, rtol=0.0001)
    #

if __name__ == '__main__':
    unittest.main()

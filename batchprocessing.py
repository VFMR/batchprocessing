from functools import wraps
import os
import shutil
import json
from typing import Union
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import pandas as pd


class BatchProcessor:
    def __init__(self,
                 n_batches: int = 1,
                 checkpoint_path: str = '',
                 do_load_cp: bool = True,
                 n_jobs: int = 1) -> None:
        self._n_batches = n_batches
        self._checkpoint_path = checkpoint_path
        self._do_load_cp = do_load_cp
        self._n_jobs = n_jobs

    def batch_predict(self, method):
        @wraps(method)
        def _wrapper(predictor_self, *args, **kwargs):
            output = self._batch_predict_func(predictor_self,
                                              method,
                                              args,
                                              kwargs)
            return output
        return _wrapper

    def _batch_predict_func(self, predictor_self, method, args, kwargs):
        if self._n_batches is None or self._n_batches == 1:
            # execute normally
            output = method(predictor_self, *args, **kwargs)
        else:
            # batch processing:
            batches, frst_it = self._get_remaining_batches_and_iterator(kwargs)
            self._check_makedir()

            last_iter = self._get_last_iter(kwargs)
            other_kwargs = self._get_other_kwargs(kwargs)

            # execute function for each batch
            if self._n_jobs is not None and self._n_jobs > 1:
                Parallel(n_jobs=self._n_jobs)(
                    delayed(self._iterfunc)(
                        predictor_self=predictor_self,
                        x=x,
                        method=method,
                        args=args,
                        other_kwargs=other_kwargs,
                        i=i+frst_it,
                    ) for i, x in enumerate(tqdm(batches)))
            else:
                for i, x in enumerate(tqdm(batches)):
                    self._iterfunc(predictor_self=predictor_self,
                                   x=x,
                                   method=method,
                                   args=args,
                                   other_kwargs=other_kwargs,
                                   i=i+frst_it,
                                   )

            # combine individual batch results into one matrix
            # TODO: implement a way to combine individual results when function
            # returns multiple values (tuple)
            last_iter = self._get_last_iter(kwargs)
            results = self._load_result_checkpoints(last_iter=last_iter)
            output = pd.concat(results, axis=0, ignore_index=True)

            self._cleanup_checkpoints()

        return output

    def _get_remaining_batches_and_iterator(self, kwargs):
        batches = self._get_batches(kwargs, n_batches=self._n_batches)
        other_kwargs = {
                key: value for key, value in kwargs.items() if key != 'X'
                }

        if self._do_load_cp:
            last_iter = self._get_last_iter(parameter_dict=other_kwargs)
        else:
            last_iter = None

        if last_iter is None:
            first_iter = 0
        else:
            first_iter = last_iter + 1
            batches = self._get_unprocessed_batches(last_iter, batches)

        return batches, first_iter

    def _iterfunc(self,
                  predictor_self,
                  x,
                  method,
                  args,
                  other_kwargs,
                  i,
                  ):
        iter_output = method(predictor_self, *args, X=x, **other_kwargs)
        self._save_checkpoints(iteration=i,
                               df=iter_output,
                               parameter_dict=other_kwargs)

    @staticmethod
    def _get_unprocessed_batches(last_iter, batches):
        return batches[last_iter+1:]

    @staticmethod
    def _get_batches(kwargs: dict, n_batches: int) -> np.ndarray:
        data = kwargs['X']
        batches = np.array_split(data, n_batches)
        return batches

    @staticmethod
    def _get_other_kwargs(kwargs):
        return {key: value for key, value in kwargs.items() if key != 'X'}

    def _get_last_iter(self, parameter_dict):
        checkpoint = {}
        if self._checkpoint_path is not None:
            try:
                with open(
                        os.path.join(
                            self._checkpoint_path,
                            'checkpoint.json'),
                        'r', encoding='utf8') as f:
                    checkpoint = json.loads(f.read())
                if isinstance(checkpoint.get('last_iter'), int):
                    cp_found = True
                else:
                    print(f'Incompatible checkpoint value {checkpoint}')
                    print('Starting from beginning')
                    cp_found = False
            except FileNotFoundError as e:
                print(e)
                print('Checkpoint file not found, starting from beginning')
                cp_found = False
        else:
            cp_found = False

        if cp_found:
            # check if the different parameters have been used:
            if parameter_dict is not None:
                for key, value in parameter_dict.items():
                    if key in checkpoint.keys() and value != checkpoint[key]:
                        raise ValueError(f'Attempting to continue with different parameters. Loaded value of {key} is {checkpoint[key]} but you passed {value}. Aborting. Manually delete the checkpoint directory or adjust the parameters to continue.')
        else:
            checkpoint = {'last_iter': None}
            # if parameter_dict is not None:
            #     checkpoint.update(parameter_dict)

        return checkpoint['last_iter']

    def _load_result_checkpoints(self, last_iter):
        df_list = []

        if last_iter is not None:
            for iteration in range(last_iter+1):
                padded_iteration = self._get_padded_iterator(iteration)
                df_list.append(pd.read_csv(
                    os.path.join(self._checkpoint_path,
                                 f'cp_{padded_iteration}.csv.gz'),
                    index_col=0))

        return df_list

    def _get_padded_iterator(self, iteration: int):
        if not isinstance(self._n_batches, int):
            raise ValueError(f'Integer expected for the number of batches but received {type(self._n_batches)} instead.')
        digits = len(str(self._n_batches))
        iteration_padded = str(iteration).zfill(digits)
        return iteration_padded

    def _save_checkpoints(self,
                          iteration: int,
                          df: Union[pd.DataFrame, np.ndarray],
                          parameter_dict: dict = None,
                          ) -> None:

        padded_iteration = self._get_padded_iterator(iteration)

        checkpoint = {'last_iter': iteration}
        if parameter_dict is not None:
            checkpoint.update(parameter_dict)

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.to_csv(os.path.join(self._checkpoint_path,
                               f'cp_{padded_iteration}.csv.gz'))

        # export the checkpoint iterator last
        with open(os.path.join(self._checkpoint_path, 'checkpoint.json'),
                  'w',
                  encoding='utf8') as f:
            f.write(json.dumps(checkpoint))

    def _check_makedir(self) -> None:
        """Check if directory exists and creates it otherwise

        Args:
            path (str): Path to directory
        """
        if not os.path.isdir(self._checkpoint_path):
            os.makedirs(self._checkpoint_path, exist_ok=True)

    def _cleanup_checkpoints(self) -> None:
        """Delete the checkpoint folder and all of its contents

        Args:
            path (str): Checkpoint_folder
        """
        if os.path.isdir(self._checkpoint_path):
            shutil.rmtree(self._checkpoint_path)

    # -------------------------------------------------------------------------
    # classmethods:
    @classmethod
    def batch_predict_auto(cls, method):
        @wraps(method)
        def _wrapper(predictor_self, *args, **kwargs):
            checkpoint_path = kwargs.get('checkpoint_path')
            n_batches = kwargs.get('n_batches')
            n_jobs = kwargs.get('n_jobs')
            do_load_cp = kwargs.get('do_load_cp')
            instance = cls(checkpoint_path=checkpoint_path,
                           n_batches=n_batches,
                           n_jobs=n_jobs,
                           do_load_cp=do_load_cp)
            if checkpoint_path is not None:
                instance._checkpoint_path = checkpoint_path
            if n_batches is not None:
                instance._n_batches = n_batches
            if n_jobs is not None:
                instance._n_jobs = n_jobs
            if do_load_cp is not None:
                instance._do_load_cp = do_load_cp

            return instance._batch_predict_func(predictor_self,
                                                method,
                                                args,
                                                kwargs)
        return _wrapper

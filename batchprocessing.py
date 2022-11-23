from functools import wraps
import os
import shutil
import json
from typing import Tuple, Union
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np
import pandas as pd


def batch_predict(method):
    @wraps(method)
    def _wrapper(self, *args, **kwargs):

        checkpoint_path = _get_checkpoint_path(kwargs)
        n_batches = _get_n_batches(kwargs)
        n_jobs = _get_n_jobs(kwargs)
        do_load_cp = _get_do_load_cp(kwargs)

        if n_batches is None or n_batches==1:
            # no batch processing. Execute method normally.
            output = method(self, *args, **kwargs)
        else:
            # batch processing
            batches = _get_batches(kwargs, n_batches)
            other_kwargs = {key: value for key, value in kwargs.items() if key!='X'}

            if do_load_cp:
                _, last_iter = _check_load_checkpoints(checkpoint_path,
                                                       other_kwargs,
                                                       n_batches=n_batches)
            else:
                last_iter = None

            if last_iter is None:
                first_iter = 0
            else:
                first_iter = last_iter + 1
                batches = _get_unprocessed_batches(last_iter, batches)

            _check_makedir(checkpoint_path)

            # execute function for each batch
            if n_jobs is not None and n_jobs>1:
                Parallel(n_jobs=n_jobs)(
                    delayed(_iterfunc)(
                        self=self,
                        x=x,
                        method=method,
                        args=args,
                        other_kwargs=other_kwargs,
                        checkpoint_path=checkpoint_path,
                        i=i+first_iter,
                        n_batches=n_batches
                    ) for i, x in enumerate(tqdm(batches)))
            else:
                for i, x in enumerate(tqdm(batches)):
                    _iterfunc(self=self, x=x, method=method, args=args,
                              other_kwargs=other_kwargs,
                              checkpoint_path=checkpoint_path,
                              i=i+first_iter,
                              n_batches=n_batches)

            # combine individual batch results into one matrix
            # TODO: implement a way to combine individual results when function returns multiple values (tuple)
            results, last_iter = _check_load_checkpoints(checkpoint_path,
                                                        other_kwargs,
                                                        n_batches=n_batches)
            output = pd.concat(results,
                               axis=0,
                               ignore_index=True)

            _cleanup_checkpoints(checkpoint_path)

        return output

    return _wrapper


def _iterfunc(self, x, method, args, other_kwargs, checkpoint_path, i, n_batches):
    iter_output = method(self, *args, X=x, **other_kwargs)
    # results.append(iter_output)
    _save_checkpoints(checkpoint_path,
                        iteration=i,
                        df=iter_output,
                        parameter_dict=other_kwargs,
                        n_batches=n_batches)
    # return iter_output


def _get_unprocessed_batches(last_iter, batches):
    new_batches = batches[last_iter+1:]
    # for i, x in enumerate(batches):
    #     if i <= last_iter and last_iter > 0:
    #         continue
    #     else:
    #         new_batches.append(x)
    return new_batches


def _get_batches(kwargs: dict, n_batches: int) -> np.ndarray:
    data = kwargs['X']
    batches = np.array_split(data, n_batches)
    return batches


def _get_default_value(kwargs, param, default):
    x = kwargs.get(param)
    if x is None:
        x = default
    return x

def _get_do_load_cp(kwargs: dict, default: bool = True) -> bool:
    """Retrieve load-checkpoint flag from passed keyword dict.

    Args:
        kwargs (dict): Keywords arguments to the decorated function
        default (bool): Default value. Defaults to True

    Returns:
        bool: Indicator for whether checkpoints shall be loaded
    """
    return _get_default_value(kwargs, 'do_load_checkpoint', default)


def _get_checkpoint_path(kwargs: dict, default: str='') -> str:
    """Retrieve checkpoint path from passed keyword dict

    Example:
        >>> _get_checkpoint_path({'checkpoint_path': '~/path/to/checkpoints/'})
        '~/path/to/checkpoints/'
        >>> _get_checkpoint_path({'A': 'B'})
        ''

    Args:
        kwargs (dict): Keyword arguments to the decorated function
        default (str): Default value. Defaults to ''

    Returns:
        str: String with the directory for checkpoints
    """
    return _get_default_value(kwargs, 'checkpoint_path', default)


def _get_n_batches(kwargs: dict, default=1) -> int:
    """Retrieve number of batches from passed keyword dict

    Example:
        >>> _get_n_batches({'n_batches': 5})
        5
        >>> _get_n_batches({'A': 10})

    Args:
        kwargs (dict): Keywords arguments to the decorated function.
        default: value to resort to if n_batches has not been passed.

    Returns:
        int: Number of batches to split the data into and process.
    """
    return _get_default_value(kwargs, 'n_batches', default)


def _get_n_jobs(kwargs: dict, default=1) -> int:
    """Retrieve the number of parallel jobs to be run from passed keyword dict.

    Args:
        kwargs (dict): Keyword argumentds to the decorated function.
        default (int, optional): value to resort to if n_jobs has not been passed.
            Defaults to 1.

    Returns:
        int: Number of parallel processes.
    """
    return _get_default_value(kwargs, 'n_jobs', default)


def _check_load_checkpoints(checkpoint_path: str=None,
                           parameter_dict: dict=None,
                           n_batches: int = None) -> Tuple[np.ndarray, int]:
    if checkpoint_path is not None:
        results, last_iter = _load_checkpoints(checkpoint_path,
                                              parameter_dict=parameter_dict,
                                              n_batches=n_batches)
    else:
        results = []
        last_iter = None
    return results, last_iter


def _load_checkpoints(path_base: str,
                      parameter_dict: dict = None,
                      n_batches: int = None):
    df_list = []

    try:
        with open(os.path.join(path_base, 'checkpoint.json'), 'r') as f:
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

    if cp_found:
        # check if the different parameters have been used:
        if parameter_dict is not None:
            for key, value in parameter_dict.items():
                if key in checkpoint.keys() and value != checkpoint[key]:
                    raise ValueError(f'Attempting to continue with different parameters. Loaded value of {key} is {checkpoint[key]} but you passed {value}. Aborting. Manually delete the checkpoint directory or adjust the parameters to continue.')

        for iteration in range(checkpoint['last_iter']+1):
            padded_iteration = _get_padded_iterator(iteration, n_batches)
            df_list.append(pd.read_csv(os.path.join(path_base, f'cp_{padded_iteration}.csv.gz'), index_col=0))
    else:
        checkpoint = {'last_iter': None}
        if parameter_dict is not None:
            checkpoint.update(parameter_dict)

    return df_list, checkpoint['last_iter']


def _get_padded_iterator(iteration: int, n_batches: int):
    if not isinstance(n_batches, int):
        raise ValueError(f'Integer expected for the number of batches but received {type(n_batches)} instead.')
    digits = len(str(n_batches))
    iteration_padded = str(iteration).zfill(digits)
    return iteration_padded


def _save_checkpoints(path_base: str,
                     iteration: int,
                     df: Union[pd.DataFrame, np.ndarray],
                     parameter_dict: dict = None,
                     n_batches: int = None) -> None:

    padded_iteration = _get_padded_iterator(iteration, n_batches)

    checkpoint = {'last_iter': iteration}
    if parameter_dict is not None:
        checkpoint.update(parameter_dict)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.to_csv(os.path.join(path_base, f'cp_{padded_iteration}.csv.gz'))

    # export the checkpoint iterator last
    with open(os.path.join(path_base, 'checkpoint.json'), 'w') as f:
        f.write(json.dumps(checkpoint))


def _check_makedir(path: str) -> None:
    """Check if directory exists and creates it otherwise

    Args:
        path (str): Path to directory
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _cleanup_checkpoints(path: str) -> None:
    """Delete the checkpoint folder and all of its contents

    Args:
        path (str): Checkpoint_folder
    """
    if os.path.isdir(path):
        shutil.rmtree(path)


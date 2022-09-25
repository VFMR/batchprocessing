from functools import wraps
import os
import shutil
import json
from typing import Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd


def batch_predict(method):
    @wraps(method)
    def _wrapper(self, *args, **kwargs):

        checkpoint_path = _get_checkpoint_path(kwargs)
        n_batches = _get_n_batches(kwargs)

        if n_batches is None:
            # no batch processing. Execute method normally.
            output = method(self, *args, **kwargs)
        else:
            # batch processing
            batches = _get_batches(kwargs, n_batches)
            other_kwargs = {key: value for key, value in kwargs.items() if key!='X'}

            results, last_iter = _check_load_checkpoints(checkpoint_path,
                                                        other_kwargs)

            # execute function for each batch
            for i, x in enumerate(tqdm(batches)):
                if i <= last_iter and last_iter>0:
                    continue
                iter_output = method(self, *args, X=x, **other_kwargs)
                results.append(iter_output)
                _save_checkpoints(checkpoint_path,
                                  iteration=i,
                                  df=iter_output,
                                  parameter_dict=other_kwargs)

            # combine individual batch results into one matrix
            # TODO: implement a way to combine individual results when function returns multiple values (tuple)
            output = pd.concat(results,
                               axis=0,
                               ignore_index=True)

        _cleanup_checkpoints(checkpoint_path)

        return output

    return _wrapper


def _get_batches(kwargs: dict, n_batches: int) -> np.ndarray:
    data = kwargs['X']
    batches = np.array_split(data, n_batches)
    return batches


def _get_checkpoint_path(kwargs: dict) -> str:
    """Retrieve checkpoint path from passed keyword dict

    Example:
        >>> _get_checkpoint_path({'checkpoint_path': '~/path/to/checkpoints/'})
        '~/path/to/checkpoints/'
        >>> _get_checkpoint_path({'A': 'B'})
        ''

    Args:
        kwargs (dict): Keyword arguments to the decorated function

    Returns:
        str: String with the directory for checkpoints
    """
    checkpoint_path = kwargs.get('checkpoint_path')

    if checkpoint_path is None:  # use the current working dir
        checkpoint_path = ''
    return checkpoint_path


def _get_n_batches(kwargs: dict, default=25) -> int:
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
    n_batches = kwargs.get('n_batches')
    if n_batches is None:
        n_batches = default
    return kwargs.get('n_batches')


def _check_load_checkpoints(checkpoint_path: str=None,
                           parameter_dict: dict=None
                           ) -> Tuple[np.ndarray, int]:
    if checkpoint_path is not None:
        results, last_iter = _load_checkpoints(checkpoint_path,
                                              parameter_dict=parameter_dict)
    else:
        results = []
        last_iter = 0
    return results, last_iter


def _load_checkpoints(path_base: str, parameter_dict: dict=None):
    df_list = []

    try:
        with open(os.path.join(path_base, 'checkpoint.json'), 'r') as f:
            checkpoint = json.loads(f.read())
        if 'last_iter' in checkpoint.keys() and isinstance(checkpoint['last_iter'], int):
            cp_found = True
        else:
            print(f'Incompatible checkpoint value {checkpoint}')
            print('Starting from beginning')
            cp_found = False
    except FileNotFoundError as e:
        print(e)
        print('Checkpoint file not found, starting from beginning')
        cp_found = False
        checkpoint = {'last_iter': 0}
        if parameter_dict is not None:
            checkpoint.update(parameter_dict)

    if cp_found:
        # check if the different parameters have been used:
        if parameter_dict is not None:
            for key, value in parameter_dict.items():
                if key in checkpoint.keys() and value != checkpoint[key]:
                    raise ValueError(f'Attempting to continue with different parameters. Loaded value of {key} is {checkpoint[key]} but you passed {value}. Aborting. Manually delete the checkpoint directory or adjust the parameters to continue.')

        for iteration in range(checkpoint['last_iter']+1):
            df_list.append(pd.read_csv(os.path.join(path_base, f'cp_{iteration}.csv.gz'), index_col=0))

    return df_list, checkpoint['last_iter']


def _save_checkpoints(path_base: str,
                     iteration: int,
                     df: pd.DataFrame,
                     parameter_dict: dict=None) -> None:
    _check_makedir(path_base)

    checkpoint = {'last_iter': iteration}
    if parameter_dict is not None:
        checkpoint.update(parameter_dict)

    with open(os.path.join(path_base, 'checkpoint.json'), 'w') as f:
        f.write(json.dumps(checkpoint))

    df.to_csv(os.path.join(path_base, f'cp_{iteration}.csv.gz'))


def _check_makedir(path: str) -> None:
    """Check if directory exists and creates it otherwise

    Args:
        path (str): Path to directory
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def _cleanup_checkpoints(path: str) -> None:
    """Delete the checkpoint folder and all of its contents

    Args:
        path (str): Checkpoint_folder
    """
    if os.path.isdir(path):
        shutil.rmtree(path)


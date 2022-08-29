from functools import wraps
import os
import shutil
import json

from tqdm import tqdm
import numpy as np
import pandas as pd


def batch_predict(method):
    @wraps(method)
    def _wrapper(self, *args, **kwargs):
        if 'n_batches' in kwargs.keys():
            n_batches = kwargs['n_batches']
        else:
            n_batches = None

        if 'checkpoint_path' in kwargs.keys():
            checkpoint_path = kwargs['checkpoint_path']
        else:
            checkpoint_path = None

        if checkpoint_path is None:  # use the current working dir
            checkpoint_path = ''


        if n_batches is None:
            # no batch processing. Execute method normally.
            output = method(self, *args, **kwargs)
        else:
            # batch processing
            data = kwargs['X']
            batches = np.array_split(data, n_batches)
            other_kwargs = {key: value for key, value in kwargs.items() if key!='X'}

            if checkpoint_path is not None:
                results, last_iter = load_checkpoints(checkpoint_path, parameter_dict=other_kwargs)
            else:
                results = []

            # execute function for each batch
            for i, x in enumerate(tqdm(batches)):
                if i <= last_iter and last_iter>0:
                    continue
                iter_output = method(self, *args, X=x, **other_kwargs)
                results.append(iter_output)
                save_checkpoints(checkpoint_path, iteration=i, df=iter_output, parameter_dict=other_kwargs)

            # combine individual batch results into one matrix
            # TODO: implement a way to combine individual results when function returns multiple values (tuple)
            output = pd.concat(results,
                               axis=0,
                               ignore_index=True)

        cleanup_checkpoints(checkpoint_path)

        return output

    return _wrapper


def load_checkpoints(path_base: str, parameter_dict: dict=None):
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


def save_checkpoints(path_base: str,
                     iteration: int,
                     df: pd.DataFrame,
                     parameter_dict: dict=None) -> None:
    check_makedir(path_base)

    checkpoint = {'last_iter': iteration}
    if parameter_dict is not None:
        checkpoint.update(parameter_dict)

    with open(os.path.join(path_base, 'checkpoint.json'), 'w') as f:
        f.write(json.dumps(checkpoint))

    # with open(path_base+'.p', 'wb') as f:
    #     pickle.dump(iteration, f)
    df.to_csv(os.path.join(path_base, f'cp_{iteration}.csv.gz'))


def check_makedir(path: str) -> None:
    """Check if directory exists and creates it otherwise

    Args:
        path (str): Path to directory
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def cleanup_checkpoints(path: str) -> None:
    """Delete the checkpoint folder and all of its contents

    Args:
        path (str): Checkpoint_folder
    """
    if os.path.isdir(path):
        shutil.rmtree(path)

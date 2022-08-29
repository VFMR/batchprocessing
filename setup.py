from setuptools import setup

setup(
    name='batchprocessing',
    version='0.1',
    py_modules=['example'],
    author='Valentin Reich',
    license='MIT',
    install_requires=[
        'tqdm',
        'pandas',
        'numpy'
        ],
        )

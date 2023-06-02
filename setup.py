from setuptools import setup


setup(
    name='batchprocessing',
    version='0.3',
    py_modules=['batchprocessing'],
    author='Valentin Reich',
    license='MIT',
    install_requires=['tqdm', 
                      'numpy', 
                      'pandas'],
    extras_require={
        'dev': [
            'pytest',
            'black'
            ]
        }
        )

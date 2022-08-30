from setuptools import setup
import pathlib

MYDIR = pathlib.Path(__file__).parent
def read_file(input):
    with open(input, 'r') as f:
        content = f.read()
    return content

INSTALL_REQUIREMENTS = read_file(os.path.join(MYDIR, 'requirements.txt')).split()
  

setup(
    name='batchprocessing',
    version='0.1',
    py_modules=['example'],
    author='Valentin Reich',
    license='MIT',
    install_requires=INSTALL_REQUIREMENTS,
        )

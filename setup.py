""" Done with https://towardsdatascience.com/create-your-custom-python-package-that-you-can-pip-install-from-your-git-repository-f90465867893"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ml_opf',
    version='0.0.1',
    author='Thomas Wolgast',
    author_email='thomas.wolgast@uol.de',
    description='Some environments to learn the Optimal Power Flow with Reinforcement Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['ml_opf', 'ml_opf.*']),
    url='https://gitlab.uni-oldenburg.de/lazi4122/ml-opf',
    license='MIT',
    install_requires=[
        'numpy==1.18.3',
        'scipy',
        'numba',
        'matplotlib',
        'pandapower',
        'gym',
        'simbench',
    ],
)

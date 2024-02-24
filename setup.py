""" Done with https://towardsdatascience.com/create-your-custom-python-package-that-you-can-pip-install-from-your-git-repository-f90465867893"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='mlopf',
    version='0.0.1',
    author='Thomas Wolgast',
    author_email='thomas.wolgast@uol.de',
    description='Some environments to learn the Optimal Power Flow with Reinforcement Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['mlopf', 'mlopf.*']),
    url='https://gitlab.com/thomaswolgast/mlopf',
    license='MIT',
    install_requires=[
        'numpy==1.22.4',
        'scipy==1.10.1',
        'numba==0.56.4',
        'pandas==1.3.5',
        'matplotlib',
        'pandapower==2.13.1',
        'simbench==1.4.0',
        'gymnasium==0.29.0',
    ],
)

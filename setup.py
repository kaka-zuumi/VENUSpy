from setuptools import setup, find_packages

setup(
   name='venuspython',
   version='1.0.1',
   description='VENUS Initial Sampling and Molecular Dynamics',
   license="MIT",
#   long_description=long_description,
   author='Kazuumi Fujioka',
   author_email='kazuumi@hawaii.edu',
   url="https://github.com/kaka-zuumi/VENUSpy",
   packages=['venuspy'],
   install_requires=['numpy', 'scipy', 'ase'], #external packages as dependencies
   scripts=[
            'cli.py',
           ]
)

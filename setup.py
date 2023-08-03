#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

packages = find_packages(exclude=('tests', 'doc'))
provides = ['pop', ]

requires = []

install_requires = ['numpy',
                    'configobj',
                    'scipy',
                    'numba',
                    'astropy',
                    'numexpr',
                    'pybtex',
                    'nestle',
                    'h5py',
                    'tabulate',
                    'taurex',
                    'pandas',
                    'exotethys',
                    'click' ]

console_scripts = ['pop=pop.pop:main',]

entry_points = {'console_scripts': console_scripts, 
                'taurex.plugins': 'taurex_pipeline = pop'}

classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
]

# Handle versioning
version = '0.0.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pop',
      author='Quentin Changeat',
      author_email='quentin.changeat.18@ucl.ac.uk',
      license="BSD",
      version=version,
      description='Pipeline of Pipes',
      classifiers=classifiers,
      packages=packages,
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords=['exoplanet',
                'pipeline',
                'taurex',],
      entry_points=entry_points,
      provides=provides,
      requires=requires,
      install_requires=install_requires,
      extras_require={
        'Plot':  ["matplotlib"], },)